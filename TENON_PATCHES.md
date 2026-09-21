# Tenon reference voice patch

Based on speech-swift 0.0.27, a2ef1dd159b1c1b3cfbdb41b228437c3c29a44f1. Tenon pins this fork by immutable commit rather than tracking a moving branch.

The ICL reference voice path uses the non-streaming text-block then codec-block prefill used by MLX Audio 0.5.4. Preset voice generation is unchanged. The new throwing `synthesizeWithVoiceCloneICLCancellable` API checks Swift Task cancellation around reference preparation, within autoregressive generation and between evaluated decoder chunks. Existing synchronous APIs remain source compatible. Cancellation cannot interrupt a GPU kernel already executing. This is sentence-batched cloning, not reference voice token streaming.

Validation on Apple Silicon: isolated module compilation, four short-reference and four full-reference Chinese outputs with local ASR content checks, pre-cancel rejection, in-flight cancellation and a successful following synthesis. Decoder probes with frozen 12-frame and 76-frame synthetic codes cover pre/post-chunk cancellation, post-copy cancellation, Task cancellation before the second chunk, and subsequent decoder recovery. Listening approved the full-reference native voice for an experimental Tenon trial. These checks do not establish general voice fidelity or app audio lifecycle behavior.

Private recordings and transcripts are not part of this repository. Keep future upstream updates explicit: rebase these three source changes, re-run native voice and cancellation checks, then update Tenon's pinned revision. Remove the fork when upstream provides equivalent behavior and passes those checks.
