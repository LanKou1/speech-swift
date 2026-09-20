"""Small real architectures exercise both equal and unequal talker/predictor widths."""
import torch
import pytest
from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSTalkerConfig
from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSTalkerForConditionalGeneration
import convert_coreml as C

torch.set_num_threads(2)


def make_talker(hidden=64):
    torch.manual_seed(42)
    cfg = Qwen3TTSTalkerConfig(
        hidden_size=hidden, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        vocab_size=128, text_vocab_size=128, text_hidden_size=64,
        rope_scaling={"rope_type": "default", "mrope_section": [3, 3, 2], "interleaved": True},
        code_predictor_config=dict(hidden_size=64, intermediate_size=128,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            head_dim=16, vocab_size=128, num_code_groups=16),
    )
    cfg._attn_implementation = "eager"
    cfg.code_predictor_config._attn_implementation = "eager"
    return Qwen3TTSTalkerForConditionalGeneration(cfg).eval()


def masks(position, size):
    pad = torch.full((1, size), -1e4)
    pad[:, :position+1] = 0
    update = torch.zeros(1, size)
    update[:, position] = 1
    return pad, update


@pytest.mark.parametrize("hidden", [64, 128])
@torch.no_grad()
def test_talker_matches_upstream_and_returns_normalized_predictor_input(hidden):
    talker = make_talker(hidden)
    talker.model.norm.weight.copy_(torch.linspace(0.5, 2, hidden))
    w = C.CodeDecoderWrapper(talker, stateful=True, max_seq_len=1024)
    embeds = torch.randn(1, 5, hidden)
    native = talker.model(inputs_embeds=embeds, use_cache=False).last_hidden_state
    for pos in range(5):
        pad, update = masks(pos, 1024)
        logits, state = w(embeds[:, pos].unsqueeze(-1).unsqueeze(-1), torch.tensor([pos]), pad, update)
        torch.testing.assert_close(state.flatten(), native[:, pos].flatten(), atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(logits, talker.codec_head(native[:, pos:pos+1]), atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("hidden", [64, 128])
@torch.no_grad()
def test_predictor_projects_talker_width_before_transformer(hidden):
    cp = make_talker(hidden).code_predictor
    w = C.MultiCodeDecoderWrapper(cp)
    embeds = torch.randn(1, 5, hidden)
    native = cp.model(inputs_embeds=cp.small_to_mtp_projection(embeds), use_cache=False).last_hidden_state
    kc = torch.zeros(1, w.kv_dim * w.num_layers, 1, C.MCD_SEQ_LEN)
    vc = torch.zeros_like(kc)
    for pos in range(5):
        pad, update = masks(pos, C.MCD_SEQ_LEN)
        logits, state, kc, vc = w(embeds[:, pos].unsqueeze(-1).unsqueeze(-1), torch.tensor([pos]), kc, pad, update, vc)
        torch.testing.assert_close(state.flatten(), native[:, pos].flatten(), atol=2e-5, rtol=2e-5)
        expected = torch.stack([head(native[:, pos]) for head in cp.lm_head], dim=1)
        torch.testing.assert_close(logits, expected, atol=2e-5, rtol=2e-5)


@torch.no_grad()
def test_cache_boundaries_and_reset():
    talker = make_talker()
    stateful = C.CodeDecoderWrapper(talker, stateful=True, max_seq_len=1024)
    explicit = C.CodeDecoderWrapper(talker, stateful=False, max_seq_len=1024)
    kc, vc = torch.zeros_like(stateful.key_cache), torch.zeros_like(stateful.value_cache)
    for pos in (0, 1, 254, 255, 256, 1022, 1023):
        embed = torch.randn(1, 64, 1, 1)
        pad, update = masks(pos, 1024)
        previous = kc.clone()
        a = stateful(embed, torch.tensor([pos]), pad, update)
        b = explicit(embed, torch.tensor([pos]), pad, update, kc, vc)
        torch.testing.assert_close(a[0], b[0])
        kc, vc = b[2:]
        torch.testing.assert_close(stateful.key_cache, kc)
        assert torch.equal(previous[..., :pos], kc[..., :pos])
        assert torch.equal(previous[..., pos+1:], kc[..., pos+1:])
    stateful.key_cache.zero_(); stateful.value_cache.zero_()
    fresh = C.CodeDecoderWrapper(talker, stateful=True, max_seq_len=1024)
    pad, update = masks(0, 1024)
    torch.testing.assert_close(stateful(embed, torch.tensor([0]), pad, update)[0], fresh(embed, torch.tensor([0]), pad, update)[0])


@torch.no_grad()
def test_compiled_stateful_graph_at_1024_positions(tmp_path):
    import coremltools as ct
    import numpy as np
    talker = make_talker(128)
    wrapper = C.CodeDecoderWrapper(talker, stateful=True, max_seq_len=1024)
    embed = torch.randn(1, 128, 1, 1)
    pad, update = masks(0, 1024)
    traced = torch.jit.trace(wrapper, (embed, torch.tensor([0]), pad, update), check_trace=False)
    for module in (wrapper, traced):
        module.key_cache.zero_(); module.value_cache.zero_()
    dim = wrapper.num_layers * wrapper.kv_dim
    model = ct.convert(traced, minimum_deployment_target=ct.target.iOS18,
        compute_units=ct.ComputeUnit.CPU_ONLY,
        inputs=[ct.TensorType("input_embeds",shape=(1,128,1,1),dtype=np.float16),
                ct.TensorType("cache_length",shape=(1,),dtype=np.int32),
                ct.TensorType("key_padding_mask",shape=(1,1024),dtype=np.float16),
                ct.TensorType("kv_cache_update_mask",shape=(1,1024),dtype=np.float16)],
        states=[ct.StateType(ct.TensorType(shape=(1,dim,1,1024),dtype=np.float16),name=n)
                for n in ("key_cache","value_cache")],
        outputs=[ct.TensorType("logits",dtype=np.float16),ct.TensorType("hidden_states",dtype=np.float16)])
    state=model.make_state()
    for pos in (0,1,255,256,1023):
        pad,update=masks(pos,1024)
        expected=wrapper(embed,torch.tensor([pos]),pad,update)
        output=model.predict({"input_embeds":embed.numpy().astype(np.float16),
            "cache_length":np.array([pos],np.int32),"key_padding_mask":pad.numpy().astype(np.float16),
            "kv_cache_update_mask":update.numpy().astype(np.float16)},state=state)
        np.testing.assert_allclose(output["logits"],expected[0].numpy(),atol=0.005,rtol=0.005)
        np.testing.assert_allclose(output["hidden_states"],expected[1].numpy(),atol=0.005,rtol=0.005)


@pytest.mark.parametrize("length", [2, 9])
@torch.no_grad()
def test_speech_transformer_preserves_sliding_window(length):
    from qwen_tts.core.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2DecoderConfig
    from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2DecoderTransformerModel
    cfg=Qwen3TTSTokenizerV2DecoderConfig(hidden_size=32,latent_dim=32,intermediate_size=64,
        num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=4,head_dim=8,
        sliding_window=3,layer_scale_initial_scale=1.0)
    cfg._attn_implementation="eager"
    transformer=Qwen3TTSTokenizerV2DecoderTransformerModel(cfg).eval()
    wrapper=C.SpeechDecoderWrapper.__new__(C.SpeechDecoderWrapper)
    torch.nn.Module.__init__(wrapper);wrapper.pre_transformer=transformer
    x=torch.randn(1,length,32)
    expected=transformer(inputs_embeds=x,use_cache=False).last_hidden_state
    torch.testing.assert_close(wrapper._manual_transformer(x),expected,atol=2e-6,rtol=2e-6)
