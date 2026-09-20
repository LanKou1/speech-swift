"""The stateful CodeDecoder's K/V cache must be written once per prediction.

`CodeDecoderWrapper.forward` computes `nkc` from the cache it read, uses it
through all of attention, and only then stores it. Written as
`self.key_cache.mul_(0.0).add_(nkc)`, that emits two `coreml_update_state` ops:
the first zeroes the buffer mid-graph while `nkc` -- a value derived from
reading that same buffer -- is still live. See `write_state` in
convert_coreml.py for the measurement.

The wrapper only calls its talker's submodules, so the stand-in below is enough
to build the same graph: nothing is downloaded, and `qwen_tts` is not needed.
Random weights, toy widths -- the question is the shape of the generated graph.

    poetry run pytest -v test_state_writes.py
"""

from __future__ import annotations

import re
from collections import Counter

import pytest

torch = pytest.importorskip("torch", reason="torch not installed")
ct = pytest.importorskip("coremltools", reason="coremltools not installed")

import numpy as np  # noqa: E402
import torch.nn as nn  # noqa: E402

import convert_coreml as C  # noqa: E402

UPDATE = re.compile(r"coreml_update_state\(state=%([A-Za-z0-9_]+)")
READ = re.compile(r"read_state\(input=%([A-Za-z0-9_]+)")

LAYERS = 2
HIDDEN = 64
HEADS = 4
KV_HEADS = 2
HEAD_DIM = 16
SEQ = 16
STATE_NAMES = ("key_cache", "value_cache")


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        f = x.float()
        return (f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + 1e-6)
                * self.weight).to(x.dtype)


class MLP(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.up = nn.Linear(hidden, hidden * 2, bias=False)
        self.down = nn.Linear(hidden * 2, hidden, bias=False)

    def forward(self, x):
        return self.down(torch.nn.functional.silu(self.up(x)))


class Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(HIDDEN, HEADS * HEAD_DIM, bias=False)
        self.k_proj = nn.Linear(HIDDEN, KV_HEADS * HEAD_DIM, bias=False)
        self.v_proj = nn.Linear(HIDDEN, KV_HEADS * HEAD_DIM, bias=False)
        self.o_proj = nn.Linear(HEADS * HEAD_DIM, HIDDEN, bias=False)
        self.q_norm = RMSNorm(HEAD_DIM)
        self.k_norm = RMSNorm(HEAD_DIM)


class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm = RMSNorm(HIDDEN)
        self.post_attention_layernorm = RMSNorm(HIDDEN)
        self.self_attn = Attn()
        self.mlp = MLP(HIDDEN)


class TalkerConfig:
    num_attention_heads = HEADS
    num_key_value_heads = KV_HEADS
    head_dim = HEAD_DIM
    hidden_size = HIDDEN
    num_hidden_layers = LAYERS


class RotaryEmb(nn.Module):
    def __init__(self):
        super().__init__()
        self.inv_freq = 1.0 / (
            10000.0 ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM)
        )
        self.attention_scaling = 1.0


class TalkerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([Layer() for _ in range(LAYERS)])
        self.norm = RMSNorm(HIDDEN)
        self.rotary_emb = RotaryEmb()


class Talker(nn.Module):
    """Only what CodeDecoderWrapper reads off the real talker."""

    def __init__(self):
        super().__init__()
        self.model = TalkerModel()
        self.codec_head = nn.Linear(HIDDEN, 128, bias=False)
        self.config = TalkerConfig()


@pytest.fixture(scope="module")
def wrapper(monkeypatch_module):
    monkeypatch_module.setattr(C, "MAX_SEQ_LEN", SEQ)
    # The real conversion path installs this before converting; without it the
    # int cast on `cache_length` fails the same way it would in production.
    C.patch_coremltools()
    torch.manual_seed(0)
    return C.CodeDecoderWrapper(Talker().eval(), stateful=True).eval()


@pytest.fixture(scope="module")
def monkeypatch_module():
    from _pytest.monkeypatch import MonkeyPatch

    patch = MonkeyPatch()
    yield patch
    patch.undo()


def reset(module):
    for name in STATE_NAMES:
        getattr(module, name).zero_()


def example_inputs():
    padding = torch.zeros(1, SEQ)
    padding[0, 6:] = float("-inf")
    update = torch.zeros(1, SEQ)
    update[0, 5] = 1.0
    return (torch.randn(1, HIDDEN, 1, 1), torch.tensor([5]), padding, update)


def mil_text(module) -> str:
    reset(module)
    with torch.no_grad():
        traced = torch.jit.trace(module, example_inputs(), strict=False,
                                 check_trace=False)
    reset(module)
    for name in STATE_NAMES:
        getattr(traced, name).zero_()
    total_kv = module.kv_dim * module.num_layers
    program = ct.convert(
        traced,
        inputs=[
            ct.TensorType("input_embeds", shape=(1, HIDDEN, 1, 1), dtype=np.float16),
            ct.TensorType("cache_length", shape=(1,), dtype=np.int32),
            ct.TensorType("key_padding_mask", shape=(1, SEQ), dtype=np.float16),
            ct.TensorType("kv_cache_update_mask", shape=(1, SEQ), dtype=np.float16),
        ],
        states=[
            ct.StateType(
                wrapped_type=ct.TensorType(
                    shape=(1, total_kv, 1, SEQ), dtype=np.float16
                ),
                name=name,
            )
            for name in STATE_NAMES
        ],
        outputs=[
            ct.TensorType("logits", dtype=np.float16),
            ct.TensorType("hidden_states", dtype=np.float16),
        ],
        convert_to="milinternal",
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_ONLY,
    )
    reset(module)
    return str(program)


def update_counts(mil: str) -> Counter:
    counts: Counter = Counter()
    for line in mil.splitlines():
        match = UPDATE.search(line)
        if match:
            counts[match.group(1)] += 1
    return counts


def test_every_cache_is_written_exactly_once(wrapper):
    counts = update_counts(mil_text(wrapper))
    assert set(counts) == set(STATE_NAMES), (
        f"cache buffers never written: {sorted(set(STATE_NAMES) - set(counts))}"
    )
    repeated = {name: n for name, n in counts.items() if n != 1}
    assert not repeated, (
        f"written more than once per prediction: {repeated}. Write them through "
        "convert_coreml.write_state()."
    )


def test_no_cache_is_read_after_this_prediction_wrote_it(wrapper):
    written: set[str] = set()
    offenders: list[str] = []
    for line in mil_text(wrapper).splitlines():
        read = READ.search(line)
        if read and read.group(1) in written:
            offenders.append(read.group(1))
        update = UPDATE.search(line)
        if update:
            written.add(update.group(1))
    assert not offenders, (
        f"read after being written in the same prediction: {sorted(set(offenders))}"
    )


def test_the_old_spelling_really_did_write_twice(wrapper, monkeypatch):
    """Pin why write_state() exists.

    `forward` resolves write_state from module globals, so replacing it here
    reproduces exactly what the code did before.
    """

    def old_write(buffer, value):
        buffer.mul_(0.0).add_(value)
        return buffer

    monkeypatch.setattr(C, "write_state", old_write)
    counts = update_counts(mil_text(wrapper))
    assert set(counts) == set(STATE_NAMES)
    assert all(n == 2 for n in counts.values()), (
        f"expected two updates per buffer with the old spelling, got {dict(counts)}"
    )
