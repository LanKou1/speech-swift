import numpy as np
import pytest
from run_coreml import Pipeline, step_inputs, sample

@pytest.mark.parametrize("position", [0, 255, 256, 1023])
def test_position_masks(position):
    data = step_inputs(np.zeros(2048), position, 1024)
    assert data["input_embeds"].shape == (1, 2048, 1, 1)
    assert data["kv_cache_update_mask"].sum() == 1
    assert data["kv_cache_update_mask"][0, position] == 1
    assert (data["key_padding_mask"][0, :position+1] == 0).all()
    assert (data["key_padding_mask"][0, position+1:] < 0).all()

@pytest.mark.parametrize("position", [-1, 1024])
def test_cache_overflow_rejected(position):
    with pytest.raises(ValueError):
        step_inputs(np.zeros(2048), position, 1024)


def test_seeded_sampling_and_nonfinite_rejection():
    logits = np.linspace(-2, 2, 3072)
    assert sample(logits, np.random.default_rng(42)) == sample(logits, np.random.default_rng(42))
    logits[20] = np.nan
    with pytest.raises(ValueError):sample(logits, np.random.default_rng(42))


def test_reject_long_audio_before_inference():
    pipeline = Pipeline.__new__(Pipeline)
    pipeline.config = {"max_seq_len": 1024, "speech_decoder_frames": 125}
    pipeline.prompt = lambda *a: [np.zeros(2048)] * 20
    with pytest.raises(ValueError, match="SpeechDecoder"):
        pipeline.synthesize("hello", np.zeros(2048), max_frames=126)
