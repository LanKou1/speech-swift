# Qwen3-TTS 1.7B CoreML CPU validation

Measured on 2026-09-20 with an Apple M5 Pro, 48 GB unified memory, macOS 26.6.2.
The experimental bundle uses a 1024-position stateful talker, FP32 decoder
computation, FP16 caches, and a fixed 125-frame SpeechDecoder.

## Results

All 12 English sentences reached EOS. Parakeet-TDT v3 transcribed all 92
reference words without errors: **0% WER on this small smoke corpus**.
This checks intelligibility; it does not measure voice similarity or establish
multilingual or long-form quality.

| Metric | Result |
|---|---:|
| Samples / reference words | 12 / 92 |
| End-of-speech reached | 12 / 12 |
| Word errors | 0 |
| Generated audio | 29.60 s |
| Model loading | 12.85 s |
| Median generation latency | 26.07 s |
| p95 generation latency | 76.80 s |
| Total synthesis wall time | 430.01 s |
| Aggregate RTF (wall time / audio time) | 14.53 |
| Peak sampled synthesis process-tree RSS | 6.70 GiB |

These are CPU-only measurements under a concurrent speech workload, so they
are not controlled speed comparisons or GPU/Neural Engine performance claims.
The reference runner returns a complete waveform; latency is full generation
latency, not streaming time to first audio. Model loading is excluded from
latency and RTF. The first request is included in the table; excluding it gives
a warm-request median of 24.88 s and p95 of 79.21 s. RSS was sampled every five
seconds and is not a measurement of total unified-memory footprint.

## Procedure and reproducibility

- Bundle: [aufklarer/Qwen3-TTS-1.7B-CoreML](https://huggingface.co/aufklarer/Qwen3-TTS-1.7B-CoreML).
- The six compiled model artifacts were published at revision
  `9026fac4e56cf14bdfee657a44a65af998050600`; subsequent card/source/report updates
  leave those weights unchanged.
- One Python `Pipeline` load for all 12 requests, sequentially, CPU only.
- Seed 42 for every request, temperature 0.8, top-k 50, repetition penalty 1.05,
  maximum 125 frames. Each request gets a fresh talker state.
- Speaker embedding extracted from the synthetic fixture
  `Tests/Qwen3ASRTests/Resources/kokoro_continuous_stitched.wav`.
- After synthesis exited, each WAV was transcribed in a separate CPU process
  with `aufklarer/Parakeet-TDT-v3-CoreML-INT8-30s`. A pre-existing debug `speech`
  binary was used only to obtain transcripts; no ASR or debug-runtime timing
  is included in the TTS performance results.
- WER is word-level Levenshtein distance after lowercasing and removing
  punctuation, divided by the total reference word count.
- [Per-sentence results and scripts](https://huggingface.co/aufklarer/Qwen3-TTS-1.7B-CoreML/tree/main/validation/benchmark)
  record the corpus, transcripts, timings, and scoring procedure.

```sh
python bundle/validation/benchmark/synthesize.py \
  bundle bundle/source bundle/speaker_embedding.npy benchmark-output
python bundle/validation/benchmark/transcribe.py benchmark-output /path/to/speech
```

## Regression coverage

Twenty focused Python tests pass, covering equal and unequal talker/predictor
widths, normalized predictor input, trained input projection, state writes,
cache boundaries and reset, minimal cache capacity, vocoder sliding-window
attention, sampling, and frame-limit rejection. All six real-checkpoint
component comparisons pass; a separate synthetic test fills all 1024 cache
positions and verifies finite values and exact reset.

The source PR changes no Swift runtime, package definition, or existing model
artifacts. The current Swift CoreML runtime still targets 0.6B/256. GPU/ANE,
iOS, and long continuous synthesis are not validated by this CPU report.

## Per-sentence results

| Reference | Audio s | Generation s | WER |
|---|---:|---:|---:|
| Hello, how are you today? | 2.24 | 43.53 | 0% |
| The quick brown fox jumps over the lazy dog. | 2.96 | 103.38 | 0% |
| Please close the window before you leave. | 2.40 | 33.56 | 0% |
| Your appointment is tomorrow morning at nine. | 2.88 | 32.90 | 0% |
| I thought the meeting went very well. | 1.92 | 23.31 | 0% |
| Can you tell me where the station is? | 2.08 | 24.88 | 0% |
| The weather will be warm and sunny today. | 2.56 | 23.78 | 0% |
| She picked up the book and began to read. | 2.48 | 19.83 | 0% |
| We need to check the results one more time. | 2.56 | 22.53 | 0% |
| Thank you for your help with this project. | 2.32 | 55.05 | 0% |
| Take a deep breath and speak slowly. | 2.64 | 20.01 | 0% |
| This is a short test of speech synthesis. | 2.56 | 27.25 | 0% |

## Existing 0.6B bundle check

The same 12 prompts were also run against the unchanged published
`aufklarer/Qwen3-TTS-CoreML` bundle, revision
`66ca03b95a684d45e020b1d2d5c3ab34a48356f9`, with its supplied speaker embedding.
All 12 reached EOS. The Python benchmark adapter uses the actual stateful
interfaces of both decoder graphs; the old bundle's metadata describes explicit
cache inputs. The cached and published model files were not modified.

| CPU result | Existing 0.6B bundle | Experimental 1.7B bundle |
|---|---:|---:|
| EOS / samples | 12 / 12 | 12 / 12 |
| WER | 2.17% (2 / 92) | 0% (0 / 92) |
| Median full-generation latency | 15.69 s | 26.07 s |
| p95 full-generation latency | 23.90 s | 76.80 s |
| Aggregate RTF | 6.14 | 14.53 |

The two baseline word edits come entirely from Parakeet rendering “tomorrow”
as “to morrow”. This ASR round-trip score does not establish a perceptual
quality improvement. The models use different speaker embeddings, quantization,
and cache capacities; concurrent load also varied between runs. Treat these
numbers as separate smoke measurements, not a controlled performance regression
comparison. The larger FP32 export is an experimental correctness baseline and
does not replace the smaller shipped bundle.
