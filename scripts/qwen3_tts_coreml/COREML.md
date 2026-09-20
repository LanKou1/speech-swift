# Reproducing the Qwen3-TTS CoreML export

The six components are TextProjector, CodeEmbedder, MultiCodeEmbedder,
CodeDecoder, MultiCodeDecoder, and SpeechDecoder. The conversion source and
reference Python runner are distributed alongside each published model bundle.

## Environment

Use Apple Silicon, macOS 15 or later, and Python 3.11. Create a dedicated
environment and install `requirements-coreml.txt`. `requirements-lock.txt`
records the complete validated environment for exact dependency reproduction. The pinned PyTorch and
coremltools versions matter: conversion and state-update behavior can change
between toolchain versions.

```sh
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-coreml.txt
```

## Export

Run components sequentially to bound memory. Each invocation loads only the
checkpoint tensors required by that component. Never run multiple exports or
model benchmarks concurrently.

```sh
for component in TextProjector CodeEmbedder MultiCodeEmbedder CodeDecoder MultiCodeDecoder SpeechDecoder Embeddings; do
  python convert_coreml.py \
    --model-id Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    --revision fd4b254389122332181a7c3db7f27e918eec64e3 \
    --tokenizer-revision 7dd38ad4e9bad454aae9cd937d0cd577604fe229 \
    --max-seq-len 1024 --output-dir bundle \
    --only "$component" --compile || exit 1
done
```

`--decoder-precision fp32` is the default. FP16 computation failed the real
1.7B checkpoint checks at position 256, so the larger FP32 decoder is used for
validation and distribution. State tensors and external embeddings remain FP16.

`--no-stateful` exports explicit CodeDecoder cache inputs and outputs instead
of MLState. MultiCodeDecoder uses a separate 16-position explicit cache,
reinitialized for every audio frame. `--quantize-w8` enables optional 8-bit
palettization; unquantized exports are the numerical baseline. Quantized
variants require their own validation.

For 0.6B, select `Qwen/Qwen3-TTS-12Hz-0.6B-Base`, its own revision, and the desired
cache capacity. Do not reuse a 1.7B revision with a different model ID.

## Model dimensions

The 1.7B talker and speaker embedding have 2048 channels. Its code predictor
has 1024 channels. MultiCodeDecoder includes the trained input projection;
changing all 1024-valued dimensions to 2048 would produce an incorrect graph.
CodeDecoder returns the normalized final hidden state used by the upstream
predictor. MLState caches receive a single write per prediction. SpeechDecoder
preserves the upstream transformer's sliding attention window; its PyTorch
wrapper must match the original decoder before conversion proceeds.

`--max-seq-len` sizes the talker caches and masks together. Positions include
the complete text/speaker prompt plus autoregressive audio steps. This is a
cache capacity, not a guarantee of speech duration, performance, or Neural
Engine placement. Select and validate the compute route on the target device.

## Speaker preparation and reference inference

Prepare a speaker embedding from a reference recording you are authorized to
use. The embedding is model-size specific; the older 1024-dimensional 0.6B
embedding cannot be used with 1.7B.

```sh
python convert_coreml.py \
  --model-id Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --revision fd4b254389122332181a7c3db7f27e918eec64e3 \
  --tokenizer-revision 7dd38ad4e9bad454aae9cd937d0cd577604fe229 \
  --max-seq-len 1024 --only Embeddings \
  --reference-audio reference.wav --output-dir bundle
python run_coreml.py bundle "Hello, how are you today?" \
  --speaker-embedding bundle/speaker_embedding.npy --output hello.wav
```

The Python runner creates a fresh talker state for each request and accepts
`--compute cpu|gpu|ane|all`, `--language`, `--seed`, `--temperature`, and
`--max-frames`. It records duration, real-time factor (wall seconds divided by
audio seconds), frame count, and whether EOS was reached.

SpeechDecoder is a separate fixed-length graph: the default 125 frames are
10 seconds at 12.5 frames/second (1920 samples/frame at 24 kHz). The runner
rejects a larger requested frame count rather than silently discarding audio.
For longer audio, re-export SpeechDecoder and Embeddings with matching
`--speech-frames`, then validate the resulting graph and waveform. A
1024-position CodeDecoder does not automatically enlarge SpeechDecoder.

The speech-swift CoreML runtime also reads these bundle dimensions and supports
1.7B/1024 through `Qwen3TTSCoreMLModel.largeModelId` or the dedicated
`speech qwen3-tts-coreml --model aufklarer/Qwen3-TTS-1.7B-CoreML` command.
Supply the prepared embedding with `speakerEmbeddingURL` in Swift or
`--speaker-embedding speaker_embedding.npy` in the CLI. The 1.7B bundle defaults
to CPU; the separate 0.6B bundle remains the default model. See the
[Swift inference guide](https://github.com/soniqo/speech-swift/blob/main/docs/inference/qwen3-tts-inference.md)
for API details and frame limits.

## Validation

```sh
pytest -q test_coreml_variants.py test_state_writes.py test_coreml_runner.py
for component in TextProjector CodeEmbedder MultiCodeEmbedder CodeDecoder MultiCodeDecoder SpeechDecoder; do
  python validate_coreml.py fixtures --component "$component" --directory bundle --fixtures validation || exit 1
  python validate_coreml.py check --component "$component" --directory bundle --fixtures validation --compute cpu || exit 1
done
```

Fixture generation and compiled-model checks run in separate processes, so
PyTorch and CoreML copies of the large models do not coexist. The decoder
fixtures include sparse cache positions 254, 255, 256, 1022, and 1023, plus a
fresh-state reset check. Run `python validate_coreml.py sustain --component CodeDecoder --directory bundle
--fixtures validation --compute cpu` to fill all cache positions consecutively
with seeded synthetic embeddings and check state integrity and reset. These
checks do not substitute for a long continuous synthesis test. Repeat the compiled-model checks for any compute
route you intend to use. Validate generated speech with transcription and
listening in addition to numerical parity.
