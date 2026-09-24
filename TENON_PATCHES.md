# Tenon reference voice patches

Based on speech-swift 0.0.27, a2ef1dd159b1c1b3cfbdb41b228437c3c29a44f1. Tenon pins immutable commits of this fork.

## Reference conditioning and cancellation

ICL uses the non-streaming text-block then codec-block prefill from MLX Audio 0.5.4. The throwing `synthesizeWithVoiceCloneICLCancellable` API checks cancellation around reference preparation, within generation and between evaluated decoder chunks. Executing GPU kernels cannot be interrupted. This API retains sentence-batched output; the separate progressive API below can deliver earlier audio.

## Progressive cloned speech

`synthesizeWithVoiceCloneICLProgressively(..., onAudio:)` observes each 24-frame target-codec prefix while retaining the complete text prefill, reference, sampling and generation sequence. Each prefix uses the existing 300-frame decoder boundaries, 25-frame left context and reference cropping; the callback emits only samples beyond the previously emitted count. At completion the original full decode supplies the remaining tail and is also returned for comparison. The generated-prefix observer defaults to nil, leaving batch callers on the original generation path. Unknown-language fallback forwards the audio callback. Cancellation is checked around prefix decoding and callback delivery; callers must retain inference ownership until the producer actually returns, including after cancellation.

In the September 24 standalone, seeded three-fixture comparison, first packets arrived in 1.67–1.75 seconds versus 2.39–3.68 seconds for sentence batching, with a median improvement of 0.74 seconds. All complete final waveforms matched the batch outputs. Two emitted waveforms matched exactly; the third had RMS difference 9.79e-8 and peak difference 2e-6, with equal sample counts. Packet-time playback simulation found no underrun and minimum inventory margins of 0.47–0.60 seconds. Extended checks included a short greeting (0.64-second packet, exact PCM), mixed Chinese/English speech (1.805-second first packet versus 3.584-second batch output; emitted RMS difference 1.44e-7, peak 4.92e-6, equal sample count), a byte-identical decoder-prefix comparison crossing the 300-frame boundary, and cancellation in the first audio callback stopping after one packet.

These are bounded offline fixtures, not listening, live app startup, microphone or simultaneous ASR/SLM/TTS evidence. Partial-prefix edges can differ numerically from the final waveform, and repeated decoding adds compute. Generated-audio parity, packet inventory, memory, cancellation and listening must be rechecked for runtime changes. The ICL generation timer includes progressive callback decoding time; it no longer isolates token generation in this mode.

Reference prewarming was tested separately and not adopted: its median first-output saving was 0.22 seconds, below the frozen 0.5-second threshold. No public reference-preparation API remains. Reference caching and the application's model-preload completion boundary stay unchanged.

## Decoder and speaker feature corrections

The two pre-upsample convolutions must be initialized with kernel equal to stride, matching their loaded kernel-2 weights. The old kernel-4 initialization retained trimRight=2 after weight loading and discarded 2880 waveform samples per decoder invocation. This correction affects both cloned and preset voices. Streaming still retains only the real target frames after padding/context removal.

Speaker features now use 384-sample reflection padding, Slaney mel spacing and area normalization, and a magnitude epsilon, matching the Python recipe. Inputs shorter than one hop are zero-extended before reflection padding to keep a valid FFT window; normal voice references are unaffected.

ICL requests 300-frame decoder chunks and 25 preceding context frames, matching Python. Other callers retain defaults25/10. Single-pass decoding respects the actual chunkSize boundary. An optional startFrame skips chunks fully inside the discarded reference prefix while preserving original chunk offsets and context for retained chunks. It does not restart decoding at the target boundary. This removes reference-only work while preserving target PCM. Cancellation is checked for skipped and retained chunks.

## Validation and update boundary

Checkpoint-backed standalone probes in scripts/tenon_decoder_length_probe.swift, scripts/tenon_prefix_decode_probe.swift and scripts/tenon_progressive_clone_probe.swift accept external local assets, which are not bundled. Length assertions distinguish the old defect from the corrected decoder. Prefix assertions compare optimized output against full decoding followed by a sample crop, including chunk boundaries and cancellation. Compile these probes against Qwen3TTS when updating the runtime. Tenon integration also requires preset streaming, clone generation/recovery and its macOS gate.

On one original-reference fixture, corrected Swift decoding of frozen Python codes closely matched Python PCM (correlation0.99999949). Speaker embedding cosine improved0.860686 to0.999429, not exact network parity. A user preferred the corrected fresh native weather audition and heard no skipped syllables. These are scoped fixture/listening results, not general fidelity or device-readiness guarantees. Larger decoder chunks can increase memory and cancellation latency; measure full clone residency when changing chunk policy.

Private recordings and transcripts are not part of this repository. Future upstream updates must reconcile these source patches, repeat runtime and Tenon checks, then update the immutable pin. Remove the fork when upstream provides equivalent behavior and passes those checks.
