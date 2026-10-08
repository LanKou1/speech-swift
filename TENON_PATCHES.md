# Tenon reference voice patches

Based on speech-swift 0.0.27, a2ef1dd159b1c1b3cfbdb41b228437c3c29a44f1. Tenon pins immutable commits of this fork.

## Reference conditioning and cancellation

ICL uses the non-streaming text-block then codec-block prefill from MLX Audio 0.5.4. The throwing `synthesizeWithVoiceCloneICLCancellable` API checks cancellation around reference preparation, within generation and between evaluated decoder chunks. Executing GPU kernels cannot be interrupted. This API retains sentence-batched output; the separate progressive API below can deliver earlier audio.

## Progressive cloned speech

`synthesizeWithVoiceCloneICLProgressively(..., onAudio:)` observes each 24-frame target-codec prefix while retaining the complete text prefill, reference, sampling and generation sequence. Each prefix originally used the existing 300-frame decoder boundaries, 25-frame left context and reference cropping, emitting only samples beyond the previously emitted count, and at completion the original full decode supplied the remaining tail and was returned for comparison. Since Tenon #557 each emission decodes only its new frames (`decodeBounded`, below) and the call returns the tail. The generated-prefix observer defaults to nil, leaving batch callers on the original generation path. Unknown-language fallback forwards the audio callback. Cancellation is checked around prefix decoding and callback delivery; callers must retain inference ownership until the producer actually returns, including after cancellation.

In the September 24 standalone, seeded three-fixture comparison, first packets arrived in 1.67–1.75 seconds versus 2.39–3.68 seconds for sentence batching, with a median improvement of 0.74 seconds. All complete final waveforms matched the batch outputs. Two emitted waveforms matched exactly; the third had RMS difference 9.79e-8 and peak difference 2e-6, with equal sample counts. Packet-time playback simulation found no underrun and minimum inventory margins of 0.47–0.60 seconds. Extended checks included a short greeting (0.64-second packet, exact PCM), mixed Chinese/English speech (1.805-second first packet versus 3.584-second batch output; emitted RMS difference 1.44e-7, peak 4.92e-6, equal sample count), a byte-identical decoder-prefix comparison crossing the 300-frame boundary, and cancellation in the first audio callback stopping after one packet.

These are bounded offline fixtures, not listening, live app startup, microphone or simultaneous ASR/SLM/TTS evidence. Partial-prefix edges can differ numerically from the final waveform, and repeated decoding adds compute. Generated-audio parity, packet inventory, memory, cancellation and listening must be rechecked for runtime changes. The ICL generation timer includes progressive callback decoding time; it no longer isolates token generation in this mode.

Reference prewarming was tested separately and not adopted: its median first-output saving was 0.22 seconds, below the frozen 0.5-second threshold. Reference caching and the application's model-preload completion boundary stay unchanged. (Tenon #557 later added `preparedVoiceCloneReference` / `primeVoiceCloneReference`, below, for saving the prepared reference, not for prewarming.)

## Bounded clone decode, float16 decode math, quantized text embedding (Tenon #557)

The progressive path above re-decoded the reference tail and the whole generated prefix at every 24-frame emission (up to ~325 frames per call). The codec decoder's high-rate convolutional activations then set the process peak: a 0.6B Base 8-bit clone peaked ~4.6 GB above its resident weights in Tenon Cloud on a 16 GB Mac. `SpeechTokenizerDecoder` now splits its forward pass into `latent` (RVQ, pre-conv and the full-causal transformer) and `upsample` (the causal convolutions, receptive field ~12 codec frames). `decodeBounded(codes:from:total:step:)` runs `latent` on the same 300/25 chunk context as the chunked decode and `upsample` on at most `step` new frames plus 16 frames of context, compiled per shape. Both ICL paths decode through it; the progressive call now returns only its final tail (every earlier sample was emitted). `Qwen3TTSModel.cloneDecodeStep` (default 8) sets the step. Base (clone) bundles warm up the bounded decode's shape at load instead of the full decoder's [1, 16, 35]; preset bundles keep the original warm-up and decode.

When the decoder weights load as float16, `upsample` computes in float16 (`computesInFloat16`); otherwise MLX promotes the math back to float32. A text embedding stored with `scales`/`biases` loads as a `QuantizedEmbedding`. `preparedVoiceCloneReference` and `primeVoiceCloneReference` read and seed the reference cache, so a saved reference skips both the codec encoder and the speaker encoder (their lazily loaded weights are then never materialized).

Measured in Tenon Cloud with the LuLu reference (seeded production sampling, 8 kit lines, 16-bit WAVs): the bounded decode matched the original decode at 103–119 dB SNR or byte-identically, with step 8 at 97–100 dB; float16 decode math with float16 decoder weights at 54–55 dB (judged indistinguishable by blind listening); a primed reference gave byte-identical output. Reply peak 7.45–7.51 GB → 4.95 GB (bounded decode, compiled) → 4.24 GB (step 8); with float16 decoder weights, a 4-bit talker and an 8-bit text embedding, 2.73 GB. First streamed chunk 1.55 s → 1.17–1.20 s. These are one Mac's numbers on one reference.

## Decoder and speaker feature corrections

The two pre-upsample convolutions must be initialized with kernel equal to stride, matching their loaded kernel-2 weights. The old kernel-4 initialization retained trimRight=2 after weight loading and discarded 2880 waveform samples per decoder invocation. This correction affects both cloned and preset voices. Streaming still retains only the real target frames after padding/context removal.

Speaker features now use 384-sample reflection padding, Slaney mel spacing and area normalization, and a magnitude epsilon, matching the Python recipe. Inputs shorter than one hop are zero-extended before reflection padding to keep a valid FFT window; normal voice references are unaffected.

ICL requests 300-frame decoder chunks and 25 preceding context frames, matching Python. Other callers retain defaults25/10. Single-pass decoding respects the actual chunkSize boundary. An optional startFrame skips chunks fully inside the discarded reference prefix while preserving original chunk offsets and context for retained chunks. It does not restart decoding at the target boundary. This removes reference-only work while preserving target PCM. Cancellation is checked for skipped and retained chunks.

## Validation and update boundary

Checkpoint-backed standalone probes in scripts/tenon_decoder_length_probe.swift, scripts/tenon_prefix_decode_probe.swift and scripts/tenon_progressive_clone_probe.swift accept external local assets, which are not bundled. Length assertions distinguish the old defect from the corrected decoder. Prefix assertions compare optimized output against full decoding followed by a sample crop, including chunk boundaries and cancellation. Compile these probes against Qwen3TTS when updating the runtime. Tenon integration also requires preset streaming, clone generation/recovery and its macOS gate.

On one original-reference fixture, corrected Swift decoding of frozen Python codes closely matched Python PCM (correlation0.99999949). Speaker embedding cosine improved0.860686 to0.999429, not exact network parity. A user preferred the corrected fresh native weather audition and heard no skipped syllables. These are scoped fixture/listening results, not general fidelity or device-readiness guarantees. Larger decoder chunks can increase memory and cancellation latency; measure full clone residency when changing chunk policy.

Private recordings and transcripts are not part of this repository. Future upstream updates must reconcile these source patches, repeat runtime and Tenon checks, then update the immutable pin. Remove the fork when upstream provides equivalent behavior and passes those checks.
