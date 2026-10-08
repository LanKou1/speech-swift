# Local clone weight validation evidence

Date: 2026-10-08. Change base: `f2582959d79ee6ceab3515bab55aafdcf6425581`.

## Behavior

The synchronous `TTSWeightLoader.validateCloneWeightSet` API validates the
complete pair before model allocation. It checks every tensor consumed by the
current talker, code predictor, speaker encoder, decoder and reference encoder,
including shapes, packed widths, affine quantization triplets and per-component
floating-point precision. It also rejects malformed/truncated safetensors,
duplicate tensor names across shards, unsafe model dimensions, wrong model
families and missing/corrupt codec configuration.

The speech decoder loader performs the same decoder validation before mutating
its modules. Its half-math flag is derived from all required convolution weights,
not just the final convolution. Weight application, casts, transpose operations,
compilation and synthesis code are unchanged for accepted bundles.

The schema follows allocated modules. Published codec metadata has some stale
codebook fields; those fields do not override the package's fixed codec
architecture. A complete codec configuration of the correct family is still
required by the pair validator.

## Executed checks

The `WeightValidationTests` suite contains 21 tests. Metadata fixtures allocate
no MLX arrays. All builds/tests were serialized with the machine build lock.

| Source state | Tests executed | Assertion failures |
| --- | ---: | ---: |
| Validation present | 21 | 0 |
| Missing tensor guard removed | 21 | 181 |
| Decoder mixed-dtype guard removed | 21 | 2 |
| Both guards restored | 21 | 0 |

The first red run changed only the missing-tensor branch in
`WeightSchema.float` from throwing to returning. The unchanged tests failed in
three methods covering missing decoder tensors, missing talker tensors and
incomplete affine triplets. The 181 failures count assertions across those
methods, not 181 separate tests.

The second red run removed only `try s.uniformFloatType()` from
`validateDecoderWeights`. Two unchanged tests failed: a complete F16 decoder
with one F32 pre-convolution, and a complete F32 decoder with only its final
convolution changed to F16. The complete F16 and F32 controls remained in the
same denominator.

Replacing only the new all-convolution predicate with the historical final-
convolution predicate is not a discriminating red run once strict dtype
validation is present: mixed decoders have already been rejected. The executed
red run therefore targets the dtype rejection itself. The direct loader
integration must also be checked in the consuming application package.

The suite was executed in a small temporary SwiftPM harness using the actual
validator, local-configuration resolver and test source. Configuration structs
were copied without changes; model/loader type shells substituted for MLX-backed
classes because the tested entrypoints only use those types as namespaces. This
proves the standalone structural logic, not all package linking or model loads.

Separate read-only checks compiled the same validator and resolver and passed
both existing complete formats:

- Original 0.6B bundle: 8-bit affine linear weights, floating text embedding and
  speaker parameters, F32 codec decoder and reference encoder.
- Converted 0.6B bundle: 4-bit affine linear weights, independently 8-bit text
  embedding, BF16 floating model parameters, F16 decoder and F32 reference
  encoder.

Those controls read only configuration and safetensors headers/byte extents.
They did not load weight payloads into MLX, synthesize speech or play audio.

## Limits

Structural validation is not checkpoint authentication or a tensor checksum.
Arbitrary finite-value corruption with intact shapes and byte extents cannot be
identified by this API. No new inference/PCM parity, listening, performance or
peak-memory measurement was performed here. Other Base callers' warm-up remains
unmeasured. No public documentation website was changed in this bounded fork
fix.
