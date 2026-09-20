# Smart Turn: End-of-Turn Confirmation

Smart Turn v3.2 ([model docs](../models/smart-turn-v3.md)) answers one
question after the VAD reports a pause: has the speaker finished, or are they
mid-sentence? It reads the audio (prosody, pace, intonation), not a
transcript, and covers 23 languages. It is a compiled Core ML model, so it is
only available where CoreML can be imported.

## When to run it

Run it once per confirmed VAD pause, on the audio of the turn so far. It does
not need to run on every chunk: it looks at up to eight seconds of audio and
one call per pause is enough. `StreamingVADProcessor` does this when a
`TurnCompletionProvider` is attached, and so does `VoicePipeline` (see
[below](#voicepipeline)):

- pause confirmed, probability at or above `threshold`: `.speechEnded` fires
  as usual;
- pause confirmed, probability below `threshold`: the segment stays open.
  Speech resuming continues the same segment; the next pause asks again;
- silence reaches `maxSilenceDuration`: the segment ends anyway, with
  `endTime` at the pause start.

```swift
import SpeechVAD

let vad = try await SileroVADModel.fromPretrained(engine: .coreml)
let turn = try await SmartTurnModel.fromPretrained()
try turn.prewarm()

let processor = StreamingVADProcessor(
    model: vad,
    turnCompletion: turn,
    turnCompletionConfig: .default)
```

For a one-shot check on a recorded utterance, call the model directly:

```swift
let probability = try turn.turnCompleteProbability(
    audio: samples, sampleRate: sampleRate)
```

## Tuning

`TurnCompletionConfig` defaults: `threshold` 0.5, `maxSilenceDuration` 2.0 s,
`preRollDuration` 0.5 s.

- **threshold** — lower values end turns sooner and interrupt more often;
  higher values wait longer before replying. Watch
  `lastTurnCompletionProbability` on real conversations before moving it.
- **maxSilenceDuration** — the hard cap, measured from the start of the
  pause. A vetoed pause can never hold the turn longer than this. `0` disables
  the cap: a held segment then waits for speech or `flush()`.
- **preRollDuration** — audio before the VAD onset that is included in the
  classifier input, so a clipped first syllable still reaches the model.
- Silero's `minSilenceDuration` still decides when a pause is confirmed, and
  therefore when the classifier is asked.

One call takes a few milliseconds on Apple Silicon, so it adds nothing
noticeable to the pause the VAD already waited for. Call `prewarm()` after
loading so the first real pause does not pay for graph compilation.

## VoicePipeline

The speech-core voice agent takes the same model through
`VoicePipeline.setTurnCompletion(_:)`, called before `start()` (`nil`
detaches). Once attached, a VAD pause only ends the user's turn when the
probability reaches `PipelineConfig.turnCompletionThreshold`; below it the
pipeline keeps listening — speech that resumes continues the same turn — until
`PipelineConfig.turnCompletionMaxSilence` elapses. Eager STT respects the veto.

```swift
import SpeechCore
import SpeechVAD

let turn = try await SmartTurnModel.fromPretrained()
try turn.prewarm()

var config = PipelineConfig()
config.mode = .echo
config.turnCompletionThreshold = 0.5
config.turnCompletionMaxSilence = 2.0

let pipeline = VoicePipeline(stt: asr, tts: tts, vad: vad, config: config) { event in
    print(event)
}
pipeline.setTurnCompletion(turn)
pipeline.start()
```

| `TurnCompletionConfig` (processor) | `PipelineConfig` (pipeline) | Default |
|---|---|---|
| `threshold` | `turnCompletionThreshold` | 0.5 |
| `maxSilenceDuration` | `turnCompletionMaxSilence` | 2.0 s |
| `preRollDuration` | `preSpeechBufferDuration` (audio kept before the VAD onset) | 0.5 s / 0.6 s |

The classifier runs synchronously on the audio thread inside `pushAudio(_:)`,
once per pause, so `prewarm()` after loading matters here too. The pipeline's
`minSilenceDuration` decides when a pause is confirmed and the classifier is
asked. Pipeline events do not carry the probability; wrap the provider if you
want to log it. The `Examples/SpeechDemo` Echo tab and `Examples/iOSEchoDemo`
attach Smart Turn this way.

## Failure behaviour

A classifier that throws counts as "complete": the segment ends as it would
without Smart Turn. `lastTurnCompletionProbability` is `nil` after such a call
in `StreamingVADProcessor`; `VoicePipeline` logs the error and returns 1.0 to
the engine.

## CLI

```bash
speech turn utterance.wav
speech turn utterance.wav --threshold 0.7 --json
speech turn utterance.wav --model-dir ~/smart-turn-coreml   # local bundle, no download
speech vad-stream call.wav --smart-turn
speech vad-stream call.wav --smart-turn --turn-threshold 0.6 --turn-max-silence 1.5
```

`speech turn` loads the file (any sample rate), scores the last eight seconds
and prints the probability, whether it clears the threshold, and the
inference time. `--json` emits `probability`, `threshold`, `complete` and
`latency_ms`. `--model` selects another HuggingFace repo with the same layout.

`speech vad-stream --smart-turn` runs the Silero state machine with Smart Turn
confirming each pause; segments that would have been split at a mid-sentence
pause come out merged. `--turn-threshold` and `--turn-max-silence` map to the
config fields above.

## Cache and offline

`SmartTurnModel.fromPretrained(cacheDir:offlineMode:)` follows the same rules
as the other models — see [Cache & offline](cache-and-offline.md).
