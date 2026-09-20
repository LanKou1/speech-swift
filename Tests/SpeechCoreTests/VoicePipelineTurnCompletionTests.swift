import Foundation
import XCTest
@testable import SpeechCore
import AudioCommon
import CSpeechCore

// MARK: - Scripted models

/// Turn classifier that answers from a script and records what it was asked.
/// The engine calls it from the audio thread, so access is locked.
private final class ScriptedTurnModel: TurnCompletionProvider {
    private let lock = NSLock()
    private let answers: [Float]
    private var recordedAudio: [[Float]] = []
    private var recordedRates: [Int] = []
    var shouldThrow = false

    init(_ answers: [Float]) { self.answers = answers }

    var seenAudio: [[Float]] { lock.withLock { recordedAudio } }
    var seenRates: [Int] { lock.withLock { recordedRates } }
    var callCount: Int { lock.withLock { recordedAudio.count } }

    func turnCompleteProbability(audio: [Float], sampleRate: Int) throws -> Float {
        lock.lock()
        defer { lock.unlock() }
        recordedAudio.append(audio)
        recordedRates.append(sampleRate)
        if shouldThrow {
            throw AudioModelError.inferenceFailed(operation: "test", reason: "scripted failure")
        }
        let index = min(recordedAudio.count - 1, answers.count - 1)
        return answers[index]
    }
}

/// Energy VAD: a chunk with any signal is speech. Deterministic no matter how
/// often the engine resets it, unlike an index-scripted VAD.
private final class EnergyVAD: StreamingVADProvider {
    var inputSampleRate: Int { 16_000 }
    var chunkSize: Int { 512 }
    func processChunk(_ samples: [Float]) -> Float {
        samples.contains { abs($0) > 0.001 } ? 0.9 : 0.0
    }
    func resetState() {}
}

private final class StubSTT: SpeechRecognitionModel {
    var inputSampleRate: Int { 16_000 }
    func transcribe(audio: [Float], sampleRate: Int, language: String?) -> String { "hello" }
}

private final class StubTTS: SpeechGenerationModel {
    var sampleRate: Int { 24_000 }
    func generate(text: String, language: String?) async throws -> [Float] { [] }
}

/// Collects pipeline events. `speechStarted` arrives synchronously inside
/// `pushAudio`; `speechEnded` is emitted by the engine's worker thread.
private final class EventLog {
    private let lock = NSLock()
    private var started = 0
    private var ended = 0
    private var onSpeechEnded: (() -> Void)?

    var speechStarted: Int { lock.withLock { started } }
    var speechEnded: Int { lock.withLock { ended } }

    func expectSpeechEnded(_ handler: @escaping () -> Void) {
        lock.withLock { onSpeechEnded = handler }
    }

    func record(_ event: PipelineEvent) {
        switch event {
        case .speechStarted:
            lock.withLock { started += 1 }
        case .speechEnded:
            let handler: (() -> Void)? = lock.withLock {
                ended += 1
                return onSpeechEnded
            }
            handler?()
        default:
            break
        }
    }
}

private let sampleRate = 16_000
private let chunkSize = 512

/// Whole 512-sample chunks covering `seconds` at `amplitude` (0 = silence).
private func audio(seconds: Float, amplitude: Float) -> [Float] {
    let chunks = Int((seconds * Float(sampleRate) / Float(chunkSize)).rounded())
    return [Float](repeating: amplitude, count: chunks * chunkSize)
}

private func speech(_ seconds: Float) -> [Float] { audio(seconds: seconds, amplitude: 0.1) }
private func silence(_ seconds: Float) -> [Float] { audio(seconds: seconds, amplitude: 0) }

// MARK: - Config and vtable bridge

final class TurnCompletionBridgeTests: XCTestCase {

    func testPipelineConfigDefaults() {
        let config = PipelineConfig()
        XCTAssertEqual(config.turnCompletionThreshold, 0.5)
        XCTAssertEqual(config.turnCompletionMaxSilence, 2.0)
        XCTAssertEqual(PipelineConfig.default.turnCompletionThreshold, 0.5)
        XCTAssertEqual(PipelineConfig.default.turnCompletionMaxSilence, 2.0)

        // Matches what the engine initialises when no value is set.
        let cDefaults = sc_config_default()
        XCTAssertEqual(cDefaults.turn_completion_threshold, config.turnCompletionThreshold)
        XCTAssertEqual(cDefaults.turn_completion_max_silence, config.turnCompletionMaxSilence)
    }

    func testPipelineConfigFieldsAreSettable() {
        var config = PipelineConfig()
        config.turnCompletionThreshold = 0.7
        config.turnCompletionMaxSilence = 1.25
        XCTAssertEqual(config.turnCompletionThreshold, 0.7)
        XCTAssertEqual(config.turnCompletionMaxSilence, 1.25)
    }

    /// Call the C function pointer the way the engine does.
    private func invoke(
        _ vtable: sc_turn_completion_vtable_t,
        samples: [Float],
        sampleRate: Int32
    ) -> Float {
        samples.withUnsafeBufferPointer { buffer in
            vtable.turn_complete_probability!(vtable.context, buffer.baseAddress, buffer.count, sampleRate)
        }
    }

    private func release(_ vtable: sc_turn_completion_vtable_t) {
        Unmanaged<TurnCompletionBridge>.fromOpaque(vtable.context!).release()
    }

    func testVtableForwardsAudioAndSampleRateAndReturnsProbability() {
        let model = ScriptedTurnModel([0.37, 0.81])
        let vtable = VoicePipeline.makeTurnCompletionVtable(TurnCompletionBridge(model))
        defer { release(vtable) }
        XCTAssertNotNil(vtable.context)
        XCTAssertNotNil(vtable.turn_complete_probability)

        let samples = (0..<4_096).map { Float($0) / 4_096 }
        XCTAssertEqual(invoke(vtable, samples: samples, sampleRate: 16_000), 0.37, accuracy: 1e-6)
        XCTAssertEqual(invoke(vtable, samples: samples, sampleRate: 48_000), 0.81, accuracy: 1e-6)

        XCTAssertEqual(model.callCount, 2)
        XCTAssertEqual(model.seenRates, [16_000, 48_000])
        XCTAssertEqual(model.seenAudio[0], samples)
        XCTAssertEqual(model.seenAudio[1], samples)
    }

    func testVtableFailsOpenWhenProviderThrows() {
        let model = ScriptedTurnModel([0.0])
        model.shouldThrow = true
        let vtable = VoicePipeline.makeTurnCompletionVtable(TurnCompletionBridge(model))
        defer { release(vtable) }

        XCTAssertEqual(invoke(vtable, samples: silence(0.5), sampleRate: 16_000), 1.0)
        XCTAssertEqual(model.callCount, 1)
    }

    func testVtableHandlesEmptyBuffer() {
        let model = ScriptedTurnModel([0.25])
        let vtable = VoicePipeline.makeTurnCompletionVtable(TurnCompletionBridge(model))
        defer { release(vtable) }

        let probability = vtable.turn_complete_probability!(vtable.context, nil, 0, 16_000)
        XCTAssertEqual(probability, 0.25, accuracy: 1e-6)
        XCTAssertEqual(model.seenAudio, [[]])
        XCTAssertEqual(model.seenRates, [16_000])
    }

    func testVtableFailsOpenWithoutContext() {
        let model = ScriptedTurnModel([0.0])
        let vtable = VoicePipeline.makeTurnCompletionVtable(TurnCompletionBridge(model))
        defer { release(vtable) }

        let samples = speech(0.1)
        let probability = samples.withUnsafeBufferPointer { buffer in
            vtable.turn_complete_probability!(nil, buffer.baseAddress, buffer.count, 16_000)
        }
        XCTAssertEqual(probability, 1.0)
        XCTAssertEqual(model.callCount, 0)
    }
}

// MARK: - Real engine with stub models

/// Drives the speech-core engine with stub STT/TTS and an energy VAD, so the
/// hold / silence-cap semantics are checked end to end without any download.
/// Timing is audio time (sample counts), not wall-clock, so pushing the whole
/// script at once is deterministic.
final class VoicePipelineTurnCompletionTests: XCTestCase {

    private func makePipeline(maxSilence: Float, log: EventLog) -> VoicePipeline {
        var config = PipelineConfig()
        config.mode = .transcribeOnly
        config.minSpeechDuration = 0.25
        config.minSilenceDuration = 0.1
        config.preSpeechBufferDuration = 0
        config.eagerSTT = false
        config.warmupSTT = false
        config.maxUtteranceDuration = 0
        config.turnCompletionThreshold = 0.5
        config.turnCompletionMaxSilence = maxSilence
        return VoicePipeline(
            stt: StubSTT(), tts: StubTTS(), vad: EnergyVAD(), config: config,
            onEvent: { log.record($0) })
    }

    func testWithoutClassifierPauseEndsTurn() {
        let log = EventLog()
        let ended = expectation(description: "speechEnded")
        log.expectSpeechEnded { ended.fulfill() }

        let pipeline = makePipeline(maxSilence: 1.0, log: log)
        pipeline.start()
        defer { pipeline.stop() }

        pipeline.pushAudio(speech(1.0) + silence(0.5))
        wait(for: [ended], timeout: 5)
        XCTAssertEqual(log.speechStarted, 1)
        XCTAssertEqual(log.speechEnded, 1)
    }

    func testCompletePauseEndsTurn() {
        let model = ScriptedTurnModel([0.9])
        let log = EventLog()
        let ended = expectation(description: "speechEnded")
        log.expectSpeechEnded { ended.fulfill() }

        let pipeline = makePipeline(maxSilence: 1.0, log: log)
        pipeline.setTurnCompletion(model)
        pipeline.start()
        defer { pipeline.stop() }

        pipeline.pushAudio(speech(1.0) + silence(0.5))
        wait(for: [ended], timeout: 5)
        XCTAssertEqual(model.callCount, 1)
        XCTAssertEqual(model.seenRates, [sampleRate])
        XCTAssertEqual(log.speechStarted, 1)
        XCTAssertEqual(log.speechEnded, 1)
    }

    func testVetoedPauseHoldsTurnUntilMaxSilence() {
        let model = ScriptedTurnModel([0.0])
        let log = EventLog()
        let notYet = expectation(description: "no speechEnded while the turn is held")
        notYet.isInverted = true
        log.expectSpeechEnded { notYet.fulfill() }

        let pipeline = makePipeline(maxSilence: 1.0, log: log)
        pipeline.setTurnCompletion(model)
        pipeline.start()
        defer { pipeline.stop() }

        // Pause confirmed at ~1.1 s and vetoed; the hold has run 0.4 s of its 1.0 s cap.
        pipeline.pushAudio(speech(1.0) + silence(0.5))
        XCTAssertEqual(log.speechStarted, 1)
        XCTAssertEqual(model.callCount, 1, "classifier asked once per pause")
        XCTAssertEqual(model.seenRates, [sampleRate], "audio arrives at the VAD sample rate")
        let seen = model.seenAudio.first?.count ?? 0
        XCTAssertGreaterThanOrEqual(seen, sampleRate / 2, "classifier sees the turn so far")
        XCTAssertLessThanOrEqual(seen, 2 * sampleRate)
        XCTAssertEqual(pipeline.state, .listening, "held turn keeps listening")
        wait(for: [notYet], timeout: 0.3)
        XCTAssertEqual(log.speechEnded, 0)

        // Silence continues past the cap: the turn ends without asking again.
        let ended = expectation(description: "speechEnded after the silence cap")
        log.expectSpeechEnded { ended.fulfill() }
        pipeline.pushAudio(silence(0.7))
        wait(for: [ended], timeout: 5)
        XCTAssertEqual(log.speechEnded, 1)
        XCTAssertEqual(log.speechStarted, 1, "no second speechStarted for the same turn")
        XCTAssertEqual(model.callCount, 1, "the silence cap does not re-run the classifier")
    }

    func testVetoedPauseThenResumedSpeechIsOneTurn() {
        let model = ScriptedTurnModel([0.0, 0.95])
        let log = EventLog()
        let ended = expectation(description: "speechEnded")
        log.expectSpeechEnded { ended.fulfill() }

        let pipeline = makePipeline(maxSilence: 2.0, log: log)
        pipeline.setTurnCompletion(model)
        pipeline.start()
        defer { pipeline.stop() }

        // "I think..." (vetoed) "...we should go." (complete)
        pipeline.pushAudio(speech(1.0) + silence(0.5) + speech(1.0) + silence(0.5))
        wait(for: [ended], timeout: 5)
        XCTAssertEqual(log.speechStarted, 1, "resumed speech continues the same turn")
        XCTAssertEqual(log.speechEnded, 1)
        XCTAssertEqual(model.callCount, 2)
        let lengths = model.seenAudio.map(\.count)
        XCTAssertGreaterThan(lengths[1], lengths[0], "second call sees the whole turn")
    }

    func testThrowingClassifierEndsTurn() {
        let model = ScriptedTurnModel([0.0])
        model.shouldThrow = true
        let log = EventLog()
        let ended = expectation(description: "speechEnded")
        log.expectSpeechEnded { ended.fulfill() }

        let pipeline = makePipeline(maxSilence: 10.0, log: log)
        pipeline.setTurnCompletion(model)
        pipeline.start()
        defer { pipeline.stop() }

        pipeline.pushAudio(speech(1.0) + silence(0.5))
        wait(for: [ended], timeout: 5)
        XCTAssertEqual(model.callCount, 1)
        XCTAssertEqual(log.speechEnded, 1)
    }

    func testDetachRestoresSilenceBehaviour() {
        let model = ScriptedTurnModel([0.0])
        let log = EventLog()
        let ended = expectation(description: "speechEnded")
        log.expectSpeechEnded { ended.fulfill() }

        let pipeline = makePipeline(maxSilence: 10.0, log: log)
        pipeline.setTurnCompletion(model)
        pipeline.setTurnCompletion(nil)
        pipeline.start()
        defer { pipeline.stop() }

        pipeline.pushAudio(speech(1.0) + silence(0.5))
        wait(for: [ended], timeout: 5)
        XCTAssertEqual(model.callCount, 0, "detached classifier is never consulted")
        XCTAssertEqual(log.speechEnded, 1)
    }
}
