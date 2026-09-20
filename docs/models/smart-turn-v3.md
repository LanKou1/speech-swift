# Smart Turn v3.2 End-of-Turn Detection

## Purpose

`SmartTurnModel` estimates whether the user has finished their turn from the
audio itself: prosody, pace and intonation, not a transcript. A VAD only hears
silence, so it cannot tell a finished sentence from a pause mid-sentence. Smart
Turn runs once per confirmed VAD pause: a finished sentence gets an immediate
reply, a mid-sentence pause keeps the agent waiting.

It is not a VAD and does not detect speech on its own. Pair it with
`StreamingVADProcessor` or `VoicePipeline`, which decide when to ask.

## Source

The weights are the official
[pipecat-ai/smart-turn-v3](https://huggingface.co/pipecat-ai/smart-turn-v3)
release (version 3.2) from Daily / Pipecat,
[github.com/pipecat-ai/smart-turn](https://github.com/pipecat-ai/smart-turn).

- **License**: BSD-2-Clause

| Property | Value |
|---|---:|
| Parameters | 8.0 million |
| Input | 128,000 mono Float32 samples |
| Sample rate | 16 kHz |
| Window | 8 seconds |
| Output | 1 float, turn-complete probability in [0, 1] |
| Runtime | Compiled Core ML, CPU + Neural Engine |
| Compiled size | approximately 17 MB |
| Languages | 23: Arabic, Bengali, Chinese, Danish, Dutch, English, Finnish, French, German, Hindi, Indonesian, Italian, Japanese, Korean, Marathi, Norwegian, Polish, Portuguese, Russian, Spanish, Turkish, Ukrainian, Vietnamese |

The compiled bundle is published as
[aufklarer/Smart-Turn-v3.2-CoreML](https://huggingface.co/aufklarer/Smart-Turn-v3.2-CoreML)
(`smart_turn.mlmodelc` plus a `config.json` with
`model_type: "smart-turn-v3-coreml"`, which the loader validates).

## Architecture

Whisper-Tiny encoder, attention pooling over the encoder frames, and a small
MLP head with a sigmoid output. The compiled graph embeds the Whisper log-mel
front-end, including the zero-mean / unit-variance waveform normalisation the
upstream model was trained with, so callers pass raw 16 kHz PCM. The front-end
runs in float32 (its power spectrum overflows float16); the encoder and head
run in float16. One eight-second window takes a few milliseconds on Apple
Silicon.

## Fixed Window

The graph takes exactly eight seconds. `turnCompleteProbability(audio:sampleRate:)`
prepares the input:

- resamples to 16 kHz when the input rate differs (only the tail that can fit
  the window is resampled);
- keeps the last eight seconds of longer input;
- zero-pads shorter input at the front, so the end of the turn always sits at
  the end of the window;
- rejects non-finite samples.

Inference failures throw. The model never substitutes a probability.

## Use in StreamingVADProcessor

`StreamingVADProcessor` accepts an optional `TurnCompletionProvider`. With one
attached, a confirmed silence (`minSilenceDuration` after the offset
threshold) no longer ends the segment by itself:

1. The processor hands the classifier the turn's audio: from
   `preRollDuration` (0.5 s) before the VAD onset up to the current chunk,
   capped at the last eight seconds.
2. Probability at or above `threshold` (0.5): `.speechEnded` is emitted as
   before.
3. Below the threshold the segment is held open. The speaker resuming
   continues the same segment (no second `.speechStarted`); the next
   confirmed pause asks the classifier again with the longer turn.
4. `maxSilenceDuration` (2.0 s, measured from the start of the pause) ends a
   held segment regardless. The segment's `endTime` is where the speech
   stopped, not when the cap expired.

`flush()` settles a held segment at end of stream and `reset()` clears it. A
classifier that throws counts as "complete", so a failing model never stalls
the conversation. `isHoldingTurn` reports the hold state and
`lastTurnCompletionProbability` the most recent classifier answer.

The settings live in `TurnCompletionConfig` (`threshold`, `maxSilenceDuration`,
`preRollDuration`). Usage and tuning: [Smart Turn inference](../inference/smart-turn.md).

## Validation

`Tests/SpeechVADTests/SmartTurnTests.swift` covers:

- `SmartTurnTests` (no model download): the published constants and the
  `config.json` contract (`model_type`, 128,000-sample window, `audio` /
  `probability` feature names); window preparation (front padding, crop to the
  last eight seconds, non-finite rejection); and the processor semantics with
  a scripted VAD and classifier — a complete pause ends the segment, a vetoed
  pause followed by more speech is one segment with a single `speechStarted`,
  the silence cap ends a held turn without re-running the classifier, a brief
  blip inside a hold does not resume it, `flush()` settles a hold, a throwing
  classifier fails open, `reset()` clears the hold.
- `Tests/SpeechCoreTests/VoicePipelineTurnCompletionTests.swift` (no model
  download): `TurnCompletionBridgeTests` checks the `PipelineConfig`
  defaults and the C vtable bridge in isolation — audio and sample rate are
  forwarded, the provider's probability is returned, a throwing provider
  yields 1.0; `VoicePipelineTurnCompletionTests` drives the speech-core
  engine with stub STT/TTS/VAD and a scripted classifier — a vetoed pause
  holds the turn until `turnCompletionMaxSilence`, resumed speech is one
  turn, a complete pause ends it, a throwing classifier fails open, and
  detaching restores silence-only behaviour.
- `E2ESmartTurnTests` (downloads the model, or reads a local copy from
  `SMART_TURN_COREML_MODEL_DIR` in offline mode): a finished spoken sentence
  followed by three seconds of pause scores above 0.85 (the fp32 reference is
  0.97) and is deterministic across calls; synthetic tones and silence stay
  finite in [0, 1]; 48 kHz input lands within 0.1 of the 16 kHz result.

Upstream reports 93.7 % accuracy (fp32) on its 31.5k-clip test set. On 1,000
clips from that set the compiled Core ML bundle scores 92.9 %, identical to
the upstream fp32 graph on the same clips, and one eight-second window takes
about 3.5 ms on an Apple M5 Pro (CPU + Neural Engine).

## API

```swift
import SpeechVAD

let turn = try await SmartTurnModel.fromPretrained()
try turn.prewarm()

let probability = try turn.turnCompleteProbability(
    audio: utterance, sampleRate: 16_000)
if probability >= SmartTurnModel.defaultThreshold {
    // finished turn — reply now
}
```

Attached to the streaming VAD:

```swift
let vad = try await SileroVADModel.fromPretrained(engine: .coreml)
let turn = try await SmartTurnModel.fromPretrained()

let processor = StreamingVADProcessor(
    model: vad,
    config: .sileroDefault,
    turnCompletion: turn,
    turnCompletionConfig: TurnCompletionConfig(threshold: 0.5, maxSilenceDuration: 2.0))

for chunk in microphoneChunks {            // Float32 @ 16 kHz
    for event in processor.process(samples: chunk) {
        switch event {
        case .speechStarted(let time): print("speech at \(time)s")
        case .speechEnded(let segment): print("turn \(segment.startTime)–\(segment.endTime)s")
        }
    }
}
let remaining = processor.flush()
```

`SmartTurnModel` conforms to `TurnCompletionProvider` (`AudioCommon`), so any
other classifier with the same shape can be attached instead. The class is
only compiled where CoreML can be imported.

## CLI

```bash
speech turn utterance.wav                  # probability, complete/incomplete, latency
speech turn utterance.wav --threshold 0.7 --json
speech turn utterance.wav --model-dir ~/smart-turn-coreml   # local bundle, no download
speech vad-stream call.wav --smart-turn    # confirm each pause; mid-sentence pauses merge
speech vad-stream call.wav --smart-turn --turn-threshold 0.6 --turn-max-silence 1.5
```

## Model registry

`speech-server` lists the model as `smart-turn-v3.2-coreml` (engine
`smart-turn`, kind `turn`, aliases `smart-turn`, `smartturn`, `turn`) so it
shows up under `/v1/models`. The Realtime session protocol has no slot for it
yet.

## VoicePipeline

`VoicePipeline` (the speech-core voice agent) accepts the same classifier
through `setTurnCompletion(_:)`, which bridges `TurnCompletionProvider` to
speech-core's `sc_turn_completion_vtable_t`. Call it before `start()`; `nil`
detaches.

```swift
import SpeechCore
import SpeechVAD

let turn = try await SmartTurnModel.fromPretrained()
try turn.prewarm()

var config = PipelineConfig()
config.turnCompletionThreshold = 0.5    // default
config.turnCompletionMaxSilence = 2.0   // default, 0 = never

let pipeline = VoicePipeline(stt: asr, tts: tts, vad: vad, config: config) { event in
    // .speechEnded now means "pause confirmed and turn complete"
}
pipeline.setTurnCompletion(turn)
pipeline.start()
```

Semantics match `StreamingVADProcessor`: the engine asks once per confirmed
pause (and at the eager-STT moment) with the audio of the turn so far, at the
VAD sample rate. A probability at or above `turnCompletionThreshold` ends the
turn; below it the pipeline keeps listening, speech that resumes continues the
same turn with no second `speechStarted`, and `turnCompletionMaxSilence` of
continued silence ends the held turn anyway. Eager STT respects the veto; the
`maxUtteranceDuration` force-split bypasses the classifier. The audio before
the VAD onset comes from `preSpeechBufferDuration` (the pipeline's
counterpart of `preRollDuration`). A classifier that throws counts as
"complete" and the error is logged. Details:
[Voice pipeline](../audio/voice-pipeline.md#end-of-turn-classifier-smart-turn).
