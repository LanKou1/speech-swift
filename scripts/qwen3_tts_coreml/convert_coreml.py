#!/usr/bin/env python3
"""Convert Qwen3-TTS to 6 CoreML components.

Produces 6 CoreML models + embeddings:
  1. TextProjector.mlpackage    — text token → embedding [1, hidden_size, 1, 1]
  2. CodeEmbedder.mlpackage     — codec token → embedding [1, hidden_size, 1, 1]
  3. MultiCodeEmbedder.mlpackage — CB1-15 token → embedding [1, hidden_size, 1, 1]
  4. CodeDecoder.mlpackage      — 28-layer transformer, NCHW, scatter-write KV
  5. MultiCodeDecoder.mlpackage — 5-layer transformer, 15 lm_heads, scatter-write KV
  6. SpeechDecoder.mlpackage    — 16 codebooks → 24kHz waveform (batch T=125)
  + speaker_embedding.npy       — optional speaker x-vector (model-specific width)
  + tts_pad/bos/eos_embed.npy   — TTS special token embeddings

Architecture:
  - All models use actual PyTorch model layers (not reimplementation)
  - torch.jit.trace with monkey-patched RMSNorm for coremltools compatibility
  - NCHW layout for ANE: [1, channels, 1, seq_len]
  - W8A16 k-means palettization (optional)

Usage:
    python scripts/convert_qwen3_tts_coreml.py --model-id Qwen/Qwen3-TTS-12Hz-0.6B-Base
    python scripts/convert_qwen3_tts_coreml.py --model-id Qwen/Qwen3-TTS-12Hz-0.6B-Base --quantize-w8
"""

import os
import sys
import time
import argparse
import json
import shutil
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MAX_SEQ_LEN = 256
MCD_SEQ_LEN = 16


# ============================================================================
# State writes
# ============================================================================

def write_state(buffer, value):
    """Replace an MLState-backed buffer's contents in a single update.

    Every state buffer in this file must be written through here, and never by
    a sequence of in-place ops such as `buffer.mul_(0.0).add_(value)`.

    coremltools lowers each in-place op on a registered buffer into its own
    `coreml_update_state` and threads the next op onto the previous update's
    result. `mul_(0.0).add_(nkc)` therefore emitted, per buffer per prediction:

        %r    = read_state(state)
        %nkc  = <28 layers of attention, computed from %r>
        %zero = mul(%r, 0.0)
        %u1   = coreml_update_state(state, %zero)   # buffer := 0, mid-graph
        %sum  = add(%u1, %nkc)
        %u2   = coreml_update_state(state, %sum)

    which mutates the buffer while `%nkc` -- a value derived from reading that
    same buffer -- is still live. The Core ML CPU backend tolerates that; the
    GPU backend does not.

    That form was reproduced directly on this machine (M5 Pro, macOS 26.5,
    coremltools 9.0). A 4096-wide graph carrying it was placed entirely on the
    GPU, including its `read_state` and `write_state` ops, and the buffer it
    stored differed from the same artifact's CPU_ONLY run at cosine 0.554,
    while its output decayed with step count: 1.000, 1.000, 0.852, 0.759,
    0.718, 0.704. The identically-two-update masked-blend form -- where the
    read is consumed by the first update and nothing derived from it spans the
    mutation -- sat in the same graph and stored 0.99999914. So the hazard is
    the live read, not the update count, and this buffer has the live read: 28
    layers of attention consume the cache before `nkc` is written back.

    `buffer[:] = value` lowers to one `slice_update` feeding one
    `coreml_update_state`, with no mid-graph mutation.

    Not measured on the Qwen3-TTS graph itself, which needs the `qwen_tts`
    package and the upstream checkpoint. The argument above is structural, and
    the structure is the one that was reproduced.
    """
    buffer[:] = value
    return value


# ============================================================================
# Transformers compatibility patches
# ============================================================================

def apply_patches():
    """Fix qwen_tts on transformers 5.x."""
    import transformers
    if int(transformers.__version__.split(".")[0]) < 5:
        return
    import transformers.utils.generic
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    def _pc(*a, **kw):
        if len(a) == 1 and callable(a[0]) and not kw: return a[0]
        return lambda fn: fn
    transformers.utils.generic.check_model_inputs = _pc
    def _dri(c, device=None, **kw):
        return 1.0 / (c.rope_theta ** (
            torch.arange(0, c.head_dim, 2, dtype=torch.float32, device=device) / c.head_dim)), 1.0
    ROPE_INIT_FUNCTIONS['default'] = _dri
    from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSTalkerConfig, Qwen3TTSTalkerCodePredictorConfig
    for cls in [Qwen3TTSTalkerConfig, Qwen3TTSTalkerCodePredictorConfig]:
        orig = cls.__init__
        def _mp(o):
            def _pi(self, *a, **kw):
                kw.pop('pad_token_id', None); o(self, *a, **kw)
                if not hasattr(self, 'pad_token_id'): self.pad_token_id = None
            return _pi
        cls.__init__ = _mp(orig)


def patch_rmsnorm():
    """Monkey-patch RMSNorm to avoid dynamic dtype casts during tracing."""
    from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSRMSNorm
    def forward(self, hidden_states):
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        return self.weight.float() * hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    Qwen3TTSRMSNorm.forward = forward


def patch_coremltools():
    """Fix coremltools Int cast for multi-dimensional arrays."""
    import coremltools.converters.mil.frontend.torch.ops as ct_ops
    from coremltools.converters.mil import Builder as mb
    original_cast = ct_ops._cast
    def patched_cast(context, node, dtype, dtype_name):
        inputs = ct_ops._get_inputs(context, node, expected=1)
        x = inputs[0]
        if x.can_be_folded_to_const():
            val = x.val
            if isinstance(val, np.ndarray) and val.size == 1:
                val = val.item()
            if not isinstance(val, dtype):
                val = dtype(val)
            context.add(mb.const(val=val, name=node.name), node.name)
        elif len(x.shape) > 0:
            context.add(mb.cast(x=mb.squeeze(x=x, name=node.name+"_sq"), dtype=dtype_name, name=node.name), node.name)
        else:
            context.add(mb.cast(x=x, dtype=dtype_name, name=node.name), node.name)
    ct_ops._cast = patched_cast


def load_model(model_id, targets=None, revision=None):
    """Allocate only selected export weights; validate every selected tensor name."""
    from accelerate import init_empty_weights
    from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
    from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    config = Qwen3TTSConfig.from_pretrained(hf_hub_download(model_id, 'config.json', revision=revision))
    with init_empty_weights():
        model = Qwen3TTSForConditionalGeneration(config)
    prefixes = {
        "TextProjector": ("talker.model.text_embedding.", "talker.text_projection."),
        "CodeEmbedder": ("talker.model.codec_embedding.",),
        "MultiCodeEmbedder": ("talker.code_predictor.model.codec_embedding.",),
        "CodeDecoder": ("talker.model.layers.", "talker.model.norm.", "talker.codec_head."),
        "MultiCodeDecoder": ("talker.code_predictor.",),
        "Embeddings": ("talker.model.text_embedding.", "talker.text_projection.", "speaker_encoder."),
        "SpeechDecoder": (),
    }
    selected = tuple(p for target in (targets or prefixes) for p in prefixes[target])
    if selected:
        path = hf_hub_download(model_id, 'model.safetensors', revision=revision)
        with safe_open(path, framework="pt", device="cpu") as checkpoint:
            weights = {k: checkpoint.get_tensor(k).float() for k in checkpoint.keys() if k.startswith(selected)}
        expected = {k for k in model.state_dict() if k.startswith(selected)}
        if expected != set(weights):
            raise ValueError(f"Checkpoint mismatch: missing={expected - set(weights)}, unexpected={set(weights) - expected}")
        model.load_state_dict(weights, strict=False, assign=True)
    model.eval()
    return model


# ============================================================================
# Helper: rotate_half for RoPE
# ============================================================================

def rotate_half(x):
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


# ============================================================================
# 1. TextProjector
# ============================================================================

class TextProjectorWrapper(nn.Module):
    def __init__(self, text_embedding, text_projection):
        super().__init__()
        self.text_embedding = text_embedding
        self.text_projection = text_projection

    def forward(self, input_ids):
        emb = self.text_embedding(input_ids)           # [1, 1, 2048]
        proj = self.text_projection(emb)               # [1, 1, 1024]
        return proj.reshape(1, -1, 1, 1)  # [1, 1024, 1, 1]


# ============================================================================
# 2. CodeEmbedder
# ============================================================================

class CodeEmbedderWrapper(nn.Module):
    def __init__(self, codec_embedding):
        super().__init__()
        self.codec_embedding = codec_embedding

    def forward(self, input_ids):
        emb = self.codec_embedding(input_ids)           # [1, 1, 1024]
        return emb.reshape(1, -1, 1, 1)  # [1, 1024, 1, 1]


# ============================================================================
# 3. MultiCodeEmbedder (linearized: 15 tables × 2048 tokens)
# ============================================================================

class MultiCodeEmbedderWrapper(nn.Module):
    def __init__(self, codec_embeddings):
        super().__init__()
        # Stack all 15 tables into one: [15*2048, 1024]
        weights = torch.cat([e.weight for e in codec_embeddings], dim=0)
        self.embedding = nn.Embedding(weights.shape[0], weights.shape[1])
        self.embedding.weight = nn.Parameter(weights)

    def forward(self, input_ids):
        emb = self.embedding(input_ids)                  # [1, 1, 1024]
        return emb.reshape(1, -1, 1, 1)  # [1, 1024, 1, 1]


# ============================================================================
# 4. CodeDecoder (28-layer transformer, NCHW, scatter-write KV)
# ============================================================================

class CodeDecoderWrapper(nn.Module):
    """Wraps actual talker model layers for NCHW + scatter-write KV cache.
    Supports both stateful (MLState) and stateless (explicit I/O) conversion.
    For stateful: key_cache/value_cache are registered as buffers → ct.StateType.
    """
    def __init__(self, talker, stateful=False, max_seq_len=None):
        super().__init__()
        self.max_seq_len = MAX_SEQ_LEN if max_seq_len is None else max_seq_len
        if self.max_seq_len < 1:
            raise ValueError("max_seq_len must be positive")
        self.layers = talker.model.layers
        self.norm = talker.model.norm
        self.codec_head = talker.codec_head
        self.register_buffer('inv_freq', talker.model.rotary_emb.inv_freq.clone())
        self.attention_scaling = talker.model.rotary_emb.attention_scaling
        cfg = talker.config
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.hidden_size = cfg.hidden_size
        self.num_layers = cfg.num_hidden_layers
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.stateful = stateful
        if stateful:
            total_kv = self.kv_dim * self.num_layers
            self.register_buffer('key_cache', torch.zeros(1, total_kv, 1, self.max_seq_len))
            self.register_buffer('value_cache', torch.zeros(1, total_kv, 1, self.max_seq_len))

    def forward(self, input_embeds, cache_length, key_padding_mask, kv_cache_update_mask, key_cache=None, value_cache=None):
        if self.stateful:
            key_cache = self.key_cache
            value_cache = self.value_cache
        # RoPE from cache_length
        pos = cache_length.unsqueeze(0).float()
        inv = self.inv_freq[None, :, None].float()
        freqs = (inv @ pos.unsqueeze(-1)).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos_full = (emb.cos() * self.attention_scaling).to(input_embeds.dtype)
        sin_full = (emb.sin() * self.attention_scaling).to(input_embeds.dtype)

        hidden = input_embeds
        new_key_slices, new_value_slices = [], []
        update_mask = kv_cache_update_mask.unsqueeze(1).unsqueeze(2)

        for i, layer in enumerate(self.layers):
            start, end = i * self.kv_dim, (i + 1) * self.kv_dim
            lkc = key_cache[:, start:end, :, :]
            lvc = value_cache[:, start:end, :, :]

            residual = hidden
            h = hidden.squeeze(-1).squeeze(-1).unsqueeze(0)
            h = layer.input_layernorm(h)
            attn = layer.self_attn
            q = attn.q_proj(h).view(1, 1, self.num_heads, self.head_dim).transpose(1, 2)
            k = attn.k_proj(h).view(1, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = attn.v_proj(h).view(1, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)
            q, k = attn.q_norm(q), attn.k_norm(k)
            q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)
            cos = cos_full.unsqueeze(1); sin = sin_full.unsqueeze(1)
            q = (q * cos) + (rotate_half(q) * sin)
            k = (k * cos) + (rotate_half(k) * sin)
            nk = k.reshape(1, self.kv_dim, 1, 1)
            nv = v.reshape(1, self.kv_dim, 1, 1)
            # Write BEFORE attention
            lkc = lkc * (1.0 - update_mask) + nk * update_mask
            lvc = lvc * (1.0 - update_mask) + nv * update_mask
            kc = lkc.squeeze(2).view(1, self.num_kv_heads, self.head_dim, self.max_seq_len)
            vc = lvc.squeeze(2).view(1, self.num_kv_heads, self.head_dim, self.max_seq_len)
            n_rep = self.num_heads // self.num_kv_heads
            kc = kc.unsqueeze(2).expand(-1, -1, n_rep, -1, -1).reshape(1, self.num_heads, self.head_dim, self.max_seq_len)
            vc = vc.unsqueeze(2).expand(-1, -1, n_rep, -1, -1).reshape(1, self.num_heads, self.head_dim, self.max_seq_len)
            aw = torch.matmul(q.transpose(1, 2), kc) / (self.head_dim ** 0.5)
            aw = aw + key_padding_mask.unsqueeze(1).unsqueeze(2)
            aw = torch.softmax(aw.float(), dim=-1)
            ao = torch.matmul(aw, vc.transpose(-1, -2)).transpose(1, 2).reshape(1, 1, -1)
            ao = attn.o_proj(ao).squeeze(0).squeeze(0).unsqueeze(-1).unsqueeze(-1)
            hidden = residual + ao
            residual = hidden
            h = hidden.squeeze(-1).squeeze(-1).unsqueeze(0)
            h = layer.post_attention_layernorm(h)
            h = layer.mlp(h).squeeze(0).squeeze(0).unsqueeze(-1).unsqueeze(-1)
            hidden = residual + h
            new_key_slices.append(nk)
            new_value_slices.append(nv)

        h = hidden.squeeze(-1).squeeze(-1).unsqueeze(0)
        normed = self.norm(h)
        logits = self.codec_head(normed)
        # Upstream passes outputs.last_hidden_state (after final RMSNorm) to the predictor.
        hidden_out = normed.squeeze(0).unsqueeze(-1).unsqueeze(-1)
        new_kv_k = torch.cat(new_key_slices, dim=1)
        new_kv_v = torch.cat(new_value_slices, dim=1)
        nkc = key_cache * (1.0 - update_mask) + new_kv_k * update_mask
        nvc = value_cache * (1.0 - update_mask) + new_kv_v * update_mask
        if self.stateful:
            # One coreml_update_state per buffer; see write_state above.
            write_state(self.key_cache, nkc)
            write_state(self.value_cache, nvc)
            return logits, hidden_out
        return logits, hidden_out, nkc, nvc


# ============================================================================
# 5. MultiCodeDecoder (5-layer CP transformer, 15 lm_heads)
# ============================================================================

class MultiCodeDecoderWrapper(nn.Module):
    """Wraps code predictor for NCHW + scatter-write KV cache. 15 lm_heads output."""
    def __init__(self, code_predictor):
        super().__init__()
        self.projection = code_predictor.small_to_mtp_projection
        self.layers = code_predictor.model.layers
        self.norm = code_predictor.model.norm
        self.lm_heads = code_predictor.lm_head
        self.register_buffer('inv_freq', code_predictor.model.rotary_emb.inv_freq.clone())
        self.attention_scaling = code_predictor.model.rotary_emb.attention_scaling
        attn0 = self.layers[0].self_attn
        cfg = attn0.config
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = attn0.head_dim
        self.num_layers = len(self.layers)
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.hidden_size = cfg.hidden_size

    def forward(self, input_embeds, cache_length, key_cache, key_padding_mask, kv_cache_update_mask, value_cache):
        pos = cache_length.unsqueeze(0).float()
        inv = self.inv_freq[None, :, None].float()
        freqs = (inv @ pos.unsqueeze(-1)).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = (emb.cos() * self.attention_scaling).to(input_embeds.dtype)
        sin = (emb.sin() * self.attention_scaling).to(input_embeds.dtype)

        projected = self.projection(input_embeds.flatten(2).transpose(1, 2))
        hidden = projected.transpose(1, 2).unsqueeze(2)
        new_key_slices, new_value_slices = [], []
        update_mask = kv_cache_update_mask.unsqueeze(1).unsqueeze(2)

        for i, layer in enumerate(self.layers):
            start, end = i * self.kv_dim, (i + 1) * self.kv_dim
            lkc = key_cache[:, start:end, :, :]
            lvc = value_cache[:, start:end, :, :]

            residual = hidden
            h = hidden.squeeze(-1).squeeze(-1).unsqueeze(0)
            h = layer.input_layernorm(h)
            attn = layer.self_attn
            q = attn.q_proj(h).view(1, 1, self.num_heads, self.head_dim).transpose(1, 2)
            k = attn.k_proj(h).view(1, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = attn.v_proj(h).view(1, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)
            if hasattr(attn, 'q_norm'): q = attn.q_norm(q); k = attn.k_norm(k)
            q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)
            cos_r = cos.unsqueeze(1); sin_r = sin.unsqueeze(1)
            q = (q * cos_r) + (rotate_half(q) * sin_r)
            k = (k * cos_r) + (rotate_half(k) * sin_r)
            nk = k.reshape(1, self.kv_dim, 1, 1)
            nv = v.reshape(1, self.kv_dim, 1, 1)
            lkc = lkc * (1.0 - update_mask) + nk * update_mask
            lvc = lvc * (1.0 - update_mask) + nv * update_mask
            kc = lkc.squeeze(2).view(1, self.num_kv_heads, self.head_dim, MCD_SEQ_LEN)
            vc = lvc.squeeze(2).view(1, self.num_kv_heads, self.head_dim, MCD_SEQ_LEN)
            n_rep = self.num_heads // self.num_kv_heads
            kc = kc.unsqueeze(2).expand(-1, -1, n_rep, -1, -1).reshape(1, self.num_heads, self.head_dim, MCD_SEQ_LEN)
            vc = vc.unsqueeze(2).expand(-1, -1, n_rep, -1, -1).reshape(1, self.num_heads, self.head_dim, MCD_SEQ_LEN)
            aw = torch.matmul(q.transpose(1, 2), kc) / (self.head_dim ** 0.5)
            aw = aw + key_padding_mask.unsqueeze(1).unsqueeze(2)
            aw = torch.softmax(aw.float(), dim=-1)
            ao = torch.matmul(aw, vc.transpose(-1, -2)).transpose(1, 2).reshape(1, 1, -1)
            ao = attn.o_proj(ao).squeeze(0).squeeze(0).unsqueeze(-1).unsqueeze(-1)
            hidden = residual + ao
            residual = hidden
            h = hidden.squeeze(-1).squeeze(-1).unsqueeze(0)
            h = layer.post_attention_layernorm(h)
            h = layer.mlp(h).squeeze(0).squeeze(0).unsqueeze(-1).unsqueeze(-1)
            hidden = residual + h
            new_key_slices.append(nk)
            new_value_slices.append(nv)

        h = hidden.squeeze(-1).squeeze(-1).unsqueeze(0)
        normed = self.norm(h)
        all_logits = torch.stack([head(normed).squeeze(1) for head in self.lm_heads], dim=1)
        hidden_out = normed.squeeze(0).unsqueeze(-1).unsqueeze(-1)
        new_kv_k = torch.cat(new_key_slices, dim=1)
        new_kv_v = torch.cat(new_value_slices, dim=1)
        nkc = key_cache * (1.0 - update_mask) + new_kv_k * update_mask
        nvc = value_cache * (1.0 - update_mask) + new_kv_v * update_mask
        return all_logits, hidden_out, nkc, nvc


class TraceVQ(nn.Module):
    def __init__(self, vq_layer):
        super().__init__()
        cb = vq_layer._codebook
        emb = cb.embedding_sum / cb.cluster_usage.clamp(min=cb.epsilon)[:, None]
        self.embedding = nn.Parameter(emb, requires_grad=False)
        self.project_out = vq_layer.project_out
    def forward(self, codes):
        q = torch.nn.functional.embedding(codes, self.embedding)
        return self.project_out(q).transpose(1, 2)

class SpeechDecoderWrapper(nn.Module):
    def __init__(self, tok_model):
        super().__init__()
        d = tok_model.decoder
        self.pre_conv = d.pre_conv
        self.pre_transformer = d.pre_transformer
        self.upsample = d.upsample
        self.audio_decoder = d.decoder
        q = d.quantizer
        self.n_q_sem = q.n_q_semantic
        self.first_vq = nn.ModuleList([TraceVQ(l) for l in q.rvq_first.vq.layers])
        self.first_proj = q.rvq_first.output_proj
        self.rest_vq = nn.ModuleList([TraceVQ(l) for l in q.rvq_rest.vq.layers])
        self.rest_proj = q.rvq_rest.output_proj

    def _manual_transformer(self, h):
        pt = self.pre_transformer
        h = pt.input_proj(h)
        B, S, _ = h.shape
        pos = torch.arange(S, device=h.device).unsqueeze(0)
        rot = pt.rotary_emb
        inv = rot.inv_freq[None, :, None].float().expand(B, -1, 1)
        freqs = (inv @ pos[:, None, :].float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = (emb.cos() * rot.attention_scaling).to(h.dtype)
        sin = (emb.sin() * rot.attention_scaling).to(h.dtype)
        def _rh(x):
            return torch.cat((-x[..., x.shape[-1]//2:], x[..., :x.shape[-1]//2]), dim=-1)
        for layer in pt.layers:
            res = h
            hn = layer.input_layernorm(h)
            attn = layer.self_attn
            hs = (*hn.shape[:-1], -1, attn.head_dim)
            q = attn.q_norm(attn.q_proj(hn).view(hs)).transpose(1, 2)
            k = attn.k_norm(attn.k_proj(hn).view(hs)).transpose(1, 2)
            v = attn.v_proj(hn).view(hs).transpose(1, 2)
            q = q * cos.unsqueeze(1) + _rh(q) * sin.unsqueeze(1)
            k = k * cos.unsqueeze(1) + _rh(k) * sin.unsqueeze(1)
            sc = attn.head_dim ** -0.5
            aw = torch.matmul(q, k.transpose(-1, -2)) * sc
            positions = torch.arange(S, device=h.device)
            distance = positions[:, None] - positions[None, :]
            allowed = distance >= 0
            if layer.attention_type == "sliding_attention":
                allowed = allowed & (distance < pt.config.sliding_window)
            cm = torch.where(allowed, torch.tensor(0.0, device=h.device),
                             torch.tensor(float('-inf'), device=h.device))
            aw = torch.softmax(aw + cm.unsqueeze(0).unsqueeze(0), dim=-1, dtype=torch.float32).to(h.dtype)
            ao = torch.matmul(aw, v).transpose(1, 2).reshape(*hn.shape[:-1], -1).contiguous()
            h = res + layer.self_attn_layer_scale(attn.o_proj(ao))
            res = h
            h = res + layer.mlp_layer_scale(layer.mlp(layer.post_attention_layernorm(h)))
        return pt.output_proj(pt.norm(h))

    def forward(self, codes):
        c1 = codes[:, :self.n_q_sem, :]
        q1 = self.first_proj(self.first_vq[0](c1[:, 0, :]))
        cr = codes[:, self.n_q_sem:, :]
        qr = self.rest_vq[0](cr[:, 0, :])
        for i in range(1, len(self.rest_vq)):
            qr = qr + self.rest_vq[i](cr[:, i, :])
        h = q1 + self.rest_proj(qr)
        h = self._manual_transformer(self.pre_conv(h).transpose(1, 2)).permute(0, 2, 1)
        for blocks in self.upsample:
            for block in blocks: h = block(h)
        for block in self.audio_decoder: h = block(h)
        return h.clamp(-1, 1)


# ============================================================================
# Conversion helpers
# ============================================================================

def code_decoder_trace_inputs(hidden_size, total_kv, capacity, stateful):
    """Trace at position zero so every positive cache capacity is supported."""
    padding = torch.full((1, capacity), float('-inf'))
    padding[:, 0] = 0
    update = torch.zeros(1, capacity)
    update[:, 0] = 1
    inputs = (torch.randn(1, hidden_size, 1, 1), torch.tensor([0]), padding, update)
    if not stateful:
        inputs += (torch.zeros(1, total_kv, 1, capacity), torch.zeros(1, total_kv, 1, capacity))
    return inputs


def convert_to_coreml(wrapper, inputs, outputs, precision, quantize_w8):
    import coremltools as ct
    from coremltools.optimize.coreml import OpPalettizerConfig, OptimizationConfig, palettize_weights

    ml = ct.convert(wrapper, inputs=inputs, outputs=outputs,
                    minimum_deployment_target=ct.target.iOS18, compute_precision=precision, compute_units=ct.ComputeUnit.CPU_ONLY, skip_model_load=True)
    if quantize_w8:
        ml = palettize_weights(ml, OptimizationConfig(
            global_config=OpPalettizerConfig(mode="kmeans", nbits=8, weight_threshold=512)))
    return ml


def compile_all_mlpackages(output_dir: str, targets=None) -> None:
    """Compile every ``*.mlpackage`` in ``output_dir`` to ``*.mlmodelc`` and
    retain the local source for validation. Only compiled ``.mlmodelc``
    artifacts are distributed in the bundle.
    """
    import shutil as _sh
    import coremltools as ct
    out = Path(output_dir)
    for pkg in sorted(out.glob("*.mlpackage")):
        if targets is not None and pkg.stem not in targets:
            continue
        mlc = pkg.with_suffix(".mlmodelc")
        if mlc.exists():
            _sh.rmtree(mlc)
        print(f"  Compiling {pkg.name} → {mlc.name}")
        compiled = ct.utils.compile_model(str(pkg))
        _sh.move(str(compiled), str(mlc))
        # Preserve the source locally for inspection; publish only .mlmodelc.


def main():
    parser = argparse.ArgumentParser(description="Convert Qwen3-TTS to 6 CoreML models")
    parser.add_argument("--model-id", default="Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    parser.add_argument("--output-dir", default="models/Qwen3-TTS-CoreML")
    parser.add_argument("--quantize-w8", action="store_true", help="W8A16 k-means palettization")
    parser.add_argument("--tokenizer-revision", help="Pinned speech tokenizer revision")
    parser.add_argument("--tokenizer-id", default="Qwen/Qwen3-TTS-Tokenizer-12Hz",
                        help="HuggingFace tokenizer model ID for SpeechDecoder")
    parser.add_argument("--decoder-precision", choices=["fp16", "fp32"], default="fp32", help="CodeDecoder computation precision; fp32 avoids FP16 overflow")
    parser.add_argument("--revision", help="Pinned upstream checkpoint revision")
    parser.add_argument("--speech-frames", type=int, default=125, help="Fixed SpeechDecoder frame count")
    parser.add_argument("--max-seq-len", type=int, default=256, help="Talker cache positions, including the prompt")
    parser.add_argument("--reference-audio", help="Optional reference WAV used to extract speaker_embedding.npy")
    parser.add_argument("--no-stateful", action="store_true", help="Disable MLState (use explicit KV cache I/O)")
    parser.add_argument("--only", type=str, default=None, help="Comma-separated: TextProjector,CodeEmbedder,MultiCodeEmbedder,CodeDecoder,MultiCodeDecoder,SpeechDecoder,Embeddings")
    parser.add_argument("--compile", action="store_true",
                        help="Compile all saved .mlpackage files → .mlmodelc for distribution")
    args = parser.parse_args()

    import coremltools as ct
    if args.max_seq_len < 1 or args.speech_frames < 1:
        parser.error("--max-seq-len and --speech-frames must be positive")
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    os.makedirs(args.output_dir, exist_ok=True)

    apply_patches()
    patch_rmsnorm()
    patch_coremltools()

    print("=" * 60)
    print(f"Converting Qwen3-TTS → 6 CoreML models")
    print(f"Model: {args.model_id}")
    print(f"Output: {args.output_dir}")
    print(f"W8A16: {args.quantize_w8}")
    print("=" * 60)

    print("\nLoading model...")
    t0 = time.time()
    targets = args.only.split(",") if args.only else [
        "TextProjector", "CodeEmbedder", "MultiCodeEmbedder",
        "CodeDecoder", "MultiCodeDecoder", "SpeechDecoder", "Embeddings"]
    allowed = {"TextProjector", "CodeEmbedder", "MultiCodeEmbedder", "CodeDecoder", "MultiCodeDecoder", "SpeechDecoder", "Embeddings"}
    if set(targets) - allowed:
        parser.error(f"Unknown export components: {sorted(set(targets) - allowed)}")
    model = load_model(args.model_id, targets=targets, revision=args.revision)
    talker = model.talker
    hidden_size = talker.config.hidden_size
    max_seq_len = args.max_seq_len
    print(f"  Loaded in {time.time()-t0:.1f}s")

    fp16 = ct.precision.FLOAT16
    fp32 = ct.precision.FLOAT32
    decoder_precision = fp32 if args.decoder_precision == "fp32" else fp16

    # ── TextProjector ────────────────────────────────────────
    if "TextProjector" in targets:
        print("\n── TextProjector ──")
        w = TextProjectorWrapper(talker.model.text_embedding, talker.text_projection)
        w.eval()
        traced = torch.jit.trace(w, (torch.tensor([151644]),), strict=False)
        ml = convert_to_coreml(traced,
            inputs=[ct.TensorType("input_ids", shape=(1,), dtype=np.int32)],
            outputs=[ct.TensorType("input_embeds", dtype=np.float16)],
            precision=fp16, quantize_w8=args.quantize_w8)
        path = f"{args.output_dir}/TextProjector.mlpackage"
        ml.save(path)
        print(f"  Saved: {path}")

    # ── CodeEmbedder ─────────────────────────────────────────
    if "CodeEmbedder" in targets:
        print("\n── CodeEmbedder ──")
        w = CodeEmbedderWrapper(talker.model.codec_embedding)
        w.eval()
        traced = torch.jit.trace(w, (torch.tensor([2149]),), strict=False)
        ml = convert_to_coreml(traced,
            inputs=[ct.TensorType("input_ids", shape=(1,), dtype=np.int32)],
            outputs=[ct.TensorType("input_embeds", dtype=np.float16)],
            precision=fp16, quantize_w8=False)  # Embeddings stay FP16
        path = f"{args.output_dir}/CodeEmbedder.mlpackage"
        ml.save(path)
        print(f"  Saved: {path}")

    # ── MultiCodeEmbedder ────────────────────────────────────
    if "MultiCodeEmbedder" in targets:
        print("\n── MultiCodeEmbedder ──")
        w = MultiCodeEmbedderWrapper(talker.code_predictor.model.codec_embedding)
        w.eval()
        traced = torch.jit.trace(w, (torch.tensor([0]),), strict=False)
        ml = convert_to_coreml(traced,
            inputs=[ct.TensorType("input_ids", shape=(1,), dtype=np.int32)],
            outputs=[ct.TensorType("input_embeds", dtype=np.float16)],
            precision=fp16, quantize_w8=False)
        path = f"{args.output_dir}/MultiCodeEmbedder.mlpackage"
        ml.save(path)
        print(f"  Saved: {path}")

    # ── CodeDecoder ──────────────────────────────────────────
    if "CodeDecoder" in targets:
        print("\n── CodeDecoder ──")
        kv_dim = talker.config.num_key_value_heads * talker.config.head_dim
        total_kv = kv_dim * talker.config.num_hidden_layers
        use_stateful = not (args.no_stateful if hasattr(args, 'no_stateful') else False)
        w = CodeDecoderWrapper(talker, stateful=use_stateful, max_seq_len=max_seq_len)
        w.eval()
        test = code_decoder_trace_inputs(hidden_size, total_kv, max_seq_len, use_stateful)
        traced = torch.jit.trace(w, test, strict=False, check_trace=False)
        if use_stateful:
            for cache in (w.key_cache, w.value_cache, traced.key_cache, traced.value_cache):
                cache.zero_()

        if use_stateful:
            print("  Converting with MLState (stateful KV cache)...")
            state_inputs = [
                ct.TensorType("input_embeds", shape=(1, hidden_size, 1, 1), dtype=np.float16),
                ct.TensorType("cache_length", shape=(1,), dtype=np.int32),
                ct.TensorType("key_padding_mask", shape=(1, max_seq_len), dtype=np.float16),
                ct.TensorType("kv_cache_update_mask", shape=(1, max_seq_len), dtype=np.float16),
            ]
            states = [
                ct.StateType(
                    wrapped_type=ct.TensorType(shape=(1, total_kv, 1, max_seq_len), dtype=np.float16),
                    name="key_cache"),
                ct.StateType(
                    wrapped_type=ct.TensorType(shape=(1, total_kv, 1, max_seq_len), dtype=np.float16),
                    name="value_cache"),
            ]
            state_outputs = [
                ct.TensorType("logits", dtype=np.float16),
                ct.TensorType("hidden_states", dtype=np.float16),
            ]
            ml = ct.convert(traced, inputs=state_inputs, outputs=state_outputs,
                            states=states,
                            minimum_deployment_target=ct.target.iOS18,
                            compute_precision=decoder_precision, compute_units=ct.ComputeUnit.CPU_ONLY, skip_model_load=True)
            if args.quantize_w8:
                from coremltools.optimize.coreml import OpPalettizerConfig, OptimizationConfig, palettize_weights
                ml = palettize_weights(ml, OptimizationConfig(
                    global_config=OpPalettizerConfig(mode="kmeans", nbits=8, weight_threshold=512)))
        else:
            ml = convert_to_coreml(traced,
                inputs=[
                ct.TensorType("input_embeds", shape=(1, hidden_size, 1, 1), dtype=np.float16),
                ct.TensorType("cache_length", shape=(1,), dtype=np.int32),
                ct.TensorType("key_padding_mask", shape=(1, max_seq_len), dtype=np.float16),
                ct.TensorType("kv_cache_update_mask", shape=(1, max_seq_len), dtype=np.float16),
                ct.TensorType("key_cache", shape=(1, total_kv, 1, max_seq_len), dtype=np.float16),
                ct.TensorType("value_cache", shape=(1, total_kv, 1, max_seq_len), dtype=np.float16),
            ],
            outputs=[
                ct.TensorType("logits", dtype=np.float16),
                ct.TensorType("hidden_states", dtype=np.float16),
                ct.TensorType("new_key_cache", dtype=np.float16),
                ct.TensorType("new_value_cache", dtype=np.float16),
            ],
            precision=decoder_precision, quantize_w8=args.quantize_w8)
        path = f"{args.output_dir}/CodeDecoder.mlpackage"
        ml.save(path)
        print(f"  Saved: {path}")

    # ── MultiCodeDecoder ──────────────────────────────────────
    if "MultiCodeDecoder" in targets:
        print("\n── MultiCodeDecoder ──")
        cp = talker.code_predictor
        mcd_w = MultiCodeDecoderWrapper(cp)
        mcd_w.eval()
        mcd_kv_dim = mcd_w.kv_dim * mcd_w.num_layers  # 5120
        test_mcd = (torch.randn(1, hidden_size, 1, 1), torch.tensor([1]),
                    torch.randn(1, mcd_kv_dim, 1, MCD_SEQ_LEN),
                    torch.zeros(1, MCD_SEQ_LEN), torch.zeros(1, MCD_SEQ_LEN),
                    torch.randn(1, mcd_kv_dim, 1, MCD_SEQ_LEN))
        test_mcd[3][0, :2] = 0; test_mcd[3][0, 2:] = float('-inf')
        test_mcd[4][0, 1] = 1.0
        traced_mcd = torch.jit.trace(mcd_w, test_mcd, strict=False)
        # Use FLOAT32 precision for MCD (FP16 causes NaN from large RMSNorm weights)
        ml = convert_to_coreml(traced_mcd,
            inputs=[
                ct.TensorType("input_embeds", shape=(1, hidden_size, 1, 1), dtype=np.float16),
                ct.TensorType("cache_length", shape=(1,), dtype=np.int32),
                ct.TensorType("key_cache", shape=(1, mcd_kv_dim, 1, MCD_SEQ_LEN), dtype=np.float16),
                ct.TensorType("key_padding_mask", shape=(1, MCD_SEQ_LEN), dtype=np.float16),
                ct.TensorType("kv_cache_update_mask", shape=(1, MCD_SEQ_LEN), dtype=np.float16),
                ct.TensorType("value_cache", shape=(1, mcd_kv_dim, 1, MCD_SEQ_LEN), dtype=np.float16),
            ],
            outputs=[
                ct.TensorType("all_logits", dtype=np.float16),
                ct.TensorType("hidden_states", dtype=np.float16),
                ct.TensorType("new_key_cache", dtype=np.float16),
                ct.TensorType("new_value_cache", dtype=np.float16),
            ],
            precision=fp32, quantize_w8=args.quantize_w8)  # FP32 compute to avoid NaN
        path = f"{args.output_dir}/MultiCodeDecoder.mlpackage"
        ml.save(path)
        print(f"  Saved: {path}")

    # ── SpeechDecoder ───────────────────────────────────────────
    if "SpeechDecoder" in targets:
        print("\n── SpeechDecoder ──")
        # Load speech tokenizer (separate model)
        from huggingface_hub import snapshot_download as snap_dl
        tok_dir = snap_dl(args.tokenizer_id, revision=args.tokenizer_revision, allow_patterns=["*.json", "*.safetensors"])

        # Patch tokenizer's RMSNorm too
        from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2DecoderRMSNorm
        def _rms_fwd(self, h):
            h = h.float()
            return self.weight.float() * h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        Qwen3TTSTokenizerV2DecoderRMSNorm.forward = _rms_fwd

        from qwen_tts import Qwen3TTSTokenizer
        speech_tok = Qwen3TTSTokenizer.from_pretrained(tok_dir, device_map="cpu")
        decoder = speech_tok.model.decoder
        decoder.eval()
        print(f"  Decoder params: {sum(p.numel() for p in decoder.parameters()):,}")

        # Build trace-friendly wrapper (from FluidInference's approach)
        sd_w = SpeechDecoderWrapper(speech_tok.model)
        sd_w.eval()
        BATCH_T = args.speech_frames
        test_codes = torch.randint(0, 2048, (1, 16, BATCH_T))
        with torch.no_grad():
            ref = sd_w(test_codes)
            orig = decoder(test_codes)
        diff = (ref - orig).abs().max().item()
        print(f"  Wrapper verify: diff={diff:.6f}")
        torch.testing.assert_close(ref, orig, atol=1e-4, rtol=1e-4)

        print(f"  Tracing T={BATCH_T}...")
        traced_sd = torch.jit.trace(sd_w, (test_codes,), strict=False)
        ml = convert_to_coreml(traced_sd,
            inputs=[ct.TensorType("audio_codes", shape=(1, 16, BATCH_T), dtype=np.int32)],
            outputs=[ct.TensorType("audio", dtype=np.float16)],
            precision=fp16, quantize_w8=args.quantize_w8)
        path = f"{args.output_dir}/SpeechDecoder.mlpackage"
        ml.save(path)
        print(f"  Saved: {path}")

    # ── Embeddings (npy) ─────────────────────────────────────
    if "Embeddings" in targets:
        print("\n── Extracting embeddings ──")
        with torch.no_grad():
            text_emb = talker.model.text_embedding
            text_proj = talker.text_projection
            for name, tid in [("tts_pad_embed", 151671), ("tts_bos_embed", 151672), ("tts_eos_embed", 151673)]:
                e = text_proj(text_emb(torch.tensor([[tid]]))).squeeze().numpy()
                np.save(f"{args.output_dir}/{name}.npy", e.astype(np.float32))
                print(f"  {name}: {e.shape}")

            if args.reference_audio:
                import soundfile as sf
                import librosa
                audio, sr = sf.read(args.reference_audio, dtype="float32")
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                audio = librosa.resample(audio, orig_sr=sr, target_sr=24000)
                from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram
                mels = mel_spectrogram(torch.from_numpy(audio).unsqueeze(0), n_fft=1024,
                    num_mels=128, sampling_rate=24000, hop_size=256, win_size=1024,
                    fmin=0, fmax=12000).transpose(1, 2)
                speaker = model.speaker_encoder(mels)[0]
                np.save(f"{args.output_dir}/speaker_embedding.npy", speaker.cpu().numpy().reshape(-1).astype(np.float32))
            else:
                print("  Supply --reference-audio to prepare an optional speaker embedding")

        from huggingface_hub import hf_hub_download
        for name in ("vocab.json", "merges.txt", "tokenizer_config.json"):
            shutil.copyfile(hf_hub_download(args.model_id, name, revision=args.revision), Path(args.output_dir) / name)
        config = {
            "model_type": "qwen3_tts_coreml", "architecture": "6-model",
            "model_id": args.model_id, "revision": args.revision, "max_seq_len": max_seq_len,
            "hidden_size": hidden_size,
            "code_predictor_hidden_size": talker.code_predictor.config.hidden_size,
            "talker_kv_dim": talker.config.num_hidden_layers * talker.config.num_key_value_heads * talker.config.head_dim,
            "predictor_kv_dim": talker.code_predictor.config.num_hidden_layers * talker.code_predictor.config.num_key_value_heads * talker.code_predictor.config.head_dim,
            "tokenizer_id": args.tokenizer_id, "tokenizer_revision": args.tokenizer_revision,
            "stateful": not args.no_stateful, "sample_rate": 24000,
            "samples_per_frame": 1920, "speech_decoder_frames": args.speech_frames,
            "requires_speaker_embedding": True,
            "quantization": "W8" if args.quantize_w8 else "none",
            "compute_precision": {"CodeDecoder": args.decoder_precision, "MultiCodeDecoder": "fp32", "embeddings": "fp16", "SpeechDecoder": "fp16"},
        }
        (Path(args.output_dir) / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    # ── Compile ──────────────────────────────────────────────
    if args.compile:
        print("\n── Compiling .mlpackage → .mlmodelc ──")
        compile_all_mlpackages(args.output_dir, targets)

    # ── Summary ──────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    for f in sorted(os.listdir(args.output_dir)):
        path = os.path.join(args.output_dir, f)
        if os.path.isdir(path):
            size = sum(os.path.getsize(os.path.join(dp, fn)) for dp, _, fns in os.walk(path) for fn in fns)
        else:
            size = os.path.getsize(path)
        print(f"  {f:40s} {size/1e6:.0f} MB")
    print("=" * 60)


if __name__ == "__main__":
    main()
