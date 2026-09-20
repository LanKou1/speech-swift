# IndexTTS2

IndexTTS2 is a zero-shot voice-cloning TTS model from `IndexTeam/IndexTTS-2`.
This package ships a native Swift/MLX runtime for the expanded
`aufklarer/IndexTTS2-MLX-fp16` bundle.

## Status

| Item | Value |
|---|---|
| Module | `IndexTTS2TTS` |
| Upstream | `IndexTeam/IndexTTS-2` |
| Default bundle | `aufklarer/IndexTTS2-MLX-fp16` |
| Parameters | 1.5B-class |
| Backend | MLX |
| Sample rate | 22.05 kHz output |
| Runtime | Native reference conditioning, semantic GPT, S2Mel flow, BigVGAN |
| Streaming | Not supported |

`IndexTTS2TTSModel` conforms to `SpeechGenerationModel` so generic callers can
identify it as a TTS model. The protocol-only `generate(text:language:)` method
throws a reference-required error because IndexTTS2 is a zero-shot voice-cloning
model. Use the model-specific `generate(text:referenceAudio:...)` overload.

## Bundle Contract

The expanded bundle contains:

- `soniqo_manifest.json` with model key, source repo, sample rate, parameter
  class, license posture, converted files, and runtime status.
- Root-level IndexTTS2 GPT/S2Mel/Qwen artifacts.
- Auxiliary model weights under `aux/`.
- tokenizer/config/license files copied from upstream repositories.

`IndexTTS2BundleLoader` validates that the manifest model key matches
`indextts2` and that all converted and copied files listed in the manifest
exist. `memoryFootprint` reports the total bytes of converted safetensors.

Publishing the bundle to Hugging Face is a release operation outside this
package. The runtime only downloads or loads the published bundle.

## Auxiliary Models

`IndexTTS2TTSModel.auxiliaryModels` exposes the complete upstream runtime
surface:

| Component | Upstream repo | Purpose |
|---|---|---|
| w2v-BERT 2.0 | `facebook/w2v-bert-2.0` | SeamlessM4T features and semantic hidden states for reference audio |
| MaskGCT semantic codec | `amphion/MaskGCT` | Quantizes reference features and maps generated semantic codes to embeddings |
| CAMPPlus | `funasr/campplus` | 192-d global style vector from the speaker reference |
| BigVGAN | `nvidia/bigvgan_v2_22khz_80band_256x` | Vocoder from generated 80-band mels to waveform |

## Native Graph

The Swift runtime runs:

1. Reference audio feature extraction with SeamlessM4T-style features.
2. w2v-BERT hidden-state extraction from layer 17.
3. MaskGCT semantic quantization for the reference prompt.
4. S2Mel prompt length regulation and 22.05 kHz prompt mel extraction.
5. CAMPPlus style embedding extraction.
6. GPT semantic-code generation.
7. S2Mel CFM decoding.
8. BigVGAN waveform assembly.

Explicit emotion vectors use the exported `feat1.safetensors` speaker-style
matrix and `feat2.safetensors` emotion matrix. For each emotion bucket the
runtime picks the speaker-style row closest to the CAMPPlus style embedding,
sums the matching emotion rows by vector weight, and keeps the remaining weight
on the reference emotion.

`IndexTTS2Tokenizer` provides a native SentencePiece Unigram path for the
published `bpe.model` and mirrors the upstream text front end ahead of it:
`char_rep_map` punctuation rewriting (`。` → `.`, quotes and brackets → `'`),
a word boundary before every CJK character so the lattice emits a standalone
`▁` ahead of each one, and Latin uppercasing. Scalars without a piece become
`<unk>` with runs merged, as in SentencePiece. The vocabulary has no digit
pieces, so `IndexTTS2TextNormalizer` reads numbers out first — cardinals,
years, dates, times, decimals, percentages, fractions, money, ordinals,
negatives — in Chinese or English depending on the text. It stands in for
upstream's WeTextProcessing pass without porting its full grammars; glossary
handling is not ported.

## Validation Status

Current validation covers graph wiring, finite waveform output, longer English
ASR roundtrips, explicit emotion control, speaking-rate control, and
internal-pause capping. Upstream PyTorch numerical parity, subjective listening
quality, and the full benchmark language set still need broader checks before
treating this port as benchmark-grade.

## Remaining Work

- Glossary handling and the long tail of WeTextProcessing number grammars
  (units, ranges, roman numerals).
- Upstream numerical parity checks.
- Longer-form listening checks beyond the segmented ASR round trip.
- Public semantic sampling controls.
- Subjective listening plus ASR roundtrip checks across the benchmark languages.
