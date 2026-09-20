import Foundation
import XCTest
@testable import SpeechVAD
import AudioCommon

// MARK: - Scripted providers

/// VAD that replays a fixed probability script, one value per 512-sample chunk.
private final class ScriptedVAD: StreamingVADProvider {
    var probabilities: [Float]
    private var index = 0
    init(_ probabilities: [Float]) { self.probabilities = probabilities }
    var inputSampleRate: Int { 16_000 }
    var chunkSize: Int { 512 }
    func processChunk(_ samples: [Float]) -> Float {
        defer { index += 1 }
        return index < probabilities.count ? probabilities[index] : 0
    }
    func resetState() { index = 0 }
}

/// Turn classifier that answers from a script and records what it was asked.
private final class ScriptedTurnModel: TurnCompletionProvider {
    var answers: [Float]
    var seenLengths: [Int] = []
    var seenRates: [Int] = []
    var shouldThrow = false
    init(_ answers: [Float]) { self.answers = answers }
    func turnCompleteProbability(audio: [Float], sampleRate: Int) throws -> Float {
        seenLengths.append(audio.count)
        seenRates.append(sampleRate)
        if shouldThrow {
            throw AudioModelError.inferenceFailed(operation: "test", reason: "scripted failure")
        }
        let index = min(seenLengths.count - 1, answers.count - 1)
        return answers[index]
    }
}

private let chunkSeconds: Float = 512.0 / 16_000.0

private func chunks(_ seconds: Float) -> Int {
    Int((seconds / chunkSeconds).rounded())
}

private func script(_ segments: [(probability: Float, seconds: Float)]) -> [Float] {
    segments.flatMap { [Float](repeating: $0.probability, count: chunks($0.seconds)) }
}

private func drive(_ processor: StreamingVADProcessor, chunkCount: Int) -> [VADEvent] {
    var events = [VADEvent]()
    let audio = [Float](repeating: 0.01, count: 512)
    for _ in 0..<chunkCount {
        events.append(contentsOf: processor.process(samples: audio))
    }
    return events
}

private func segments(_ events: [VADEvent]) -> [SpeechSegment] {
    events.compactMap {
        if case .speechEnded(let segment) = $0 { return segment }
        return nil
    }
}

private func starts(_ events: [VADEvent]) -> [Float] {
    events.compactMap {
        if case .speechStarted(let time) = $0 { return time }
        return nil
    }
}

// MARK: - Unit tests

final class SmartTurnTests: XCTestCase {
    private let vadConfig = VADConfig(
        onset: 0.5, offset: 0.35,
        minSpeechDuration: 0.25, minSilenceDuration: 0.1,
        windowDuration: 0.032, stepRatio: 1.0)

    func testPublishedConfiguration() {
        #if canImport(CoreML)
        XCTAssertEqual(SmartTurnModel.defaultModelId, "aufklarer/Smart-Turn-v3.2-CoreML")
        XCTAssertEqual(SmartTurnModel.sampleRate, 16_000)
        XCTAssertEqual(SmartTurnModel.windowSeconds, 8)
        XCTAssertEqual(SmartTurnModel.windowSamples, 128_000)
        XCTAssertEqual(SmartTurnModel.defaultThreshold, 0.5)
        #endif
        XCTAssertEqual(TurnCompletionConfig.default.threshold, 0.5)
        XCTAssertEqual(TurnCompletionConfig.default.maxSilenceDuration, 2.0)
    }

    #if canImport(CoreML)
    func testDecodesPublishedModelConfiguration() throws {
        let data = Data("""
        {
          "model_type": "smart-turn-v3-coreml",
          "sample_rate": 16000,
          "window_samples": 128000,
          "input_name": "audio",
          "output_name": "probability",
          "compiled_model": "smart_turn.mlmodelc",
          "version": "3.2"
        }
        """.utf8)
        let configuration = try SmartTurnModel.decodeConfiguration(data)
        XCTAssertEqual(configuration.windowSamples, 128_000)
        XCTAssertEqual(configuration.compiledModel, "smart_turn.mlmodelc")
    }

    func testRejectsIncompatibleModelConfiguration() {
        let data = Data("""
        {
          "model_type": "smart-turn-v3-coreml",
          "sample_rate": 16000,
          "window_samples": 480000,
          "input_name": "audio",
          "output_name": "probability",
          "compiled_model": "smart_turn.mlmodelc"
        }
        """.utf8)
        XCTAssertThrowsError(try SmartTurnModel.decodeConfiguration(data))
    }

    func testPreparedWindowPadsShortTurnsAtTheFront() throws {
        let samples = (0..<1_000).map(Float.init)
        let window = try SmartTurnModel.preparedWindow(samples)
        XCTAssertEqual(window.count, 128_000)
        XCTAssertEqual(window[0], 0)
        XCTAssertEqual(window[126_999], 0)
        XCTAssertEqual(window[127_000], 0)
        XCTAssertEqual(window[127_999], 999)
    }

    func testPreparedWindowKeepsTheLastEightSeconds() throws {
        let samples = (0..<133_000).map(Float.init)
        let window = try SmartTurnModel.preparedWindow(samples)
        XCTAssertEqual(window.count, 128_000)
        XCTAssertEqual(window.first, 5_000)
        XCTAssertEqual(window.last, 132_999)
    }

    func testPreparedWindowKeepsExactWindow() throws {
        let samples = [Float](repeating: 0.25, count: 128_000)
        XCTAssertEqual(try SmartTurnModel.preparedWindow(samples), samples)
    }

    func testPreparedWindowRejectsNonFiniteSamples() {
        var samples = [Float](repeating: 0, count: 16_000)
        samples[42] = .infinity
        XCTAssertThrowsError(try SmartTurnModel.preparedWindow(samples))
    }
    #endif

    // MARK: Processor semantics

    func testWithoutClassifierBehaviourIsUnchanged() {
        let vad = ScriptedVAD(script([(0.9, 1.0), (0.0, 0.5)]))
        let processor = StreamingVADProcessor(provider: vad, config: vadConfig)
        let events = drive(processor, chunkCount: vad.probabilities.count)
        XCTAssertEqual(starts(events).count, 1)
        XCTAssertEqual(segments(events).count, 1)
        XCTAssertNil(processor.lastTurnCompletionProbability)
    }

    func testCompletePauseEndsSegment() {
        let vad = ScriptedVAD(script([(0.9, 1.0), (0.0, 0.5)]))
        let model = ScriptedTurnModel([0.93])
        let processor = StreamingVADProcessor(
            provider: vad, config: vadConfig, turnCompletion: model)
        let events = drive(processor, chunkCount: vad.probabilities.count)
        XCTAssertEqual(segments(events).count, 1)
        XCTAssertEqual(model.seenLengths.count, 1)
        XCTAssertEqual(model.seenRates, [16_000])
        XCTAssertEqual(processor.lastTurnCompletionProbability, 0.93)
        // Pre-roll plus the speech and the pending silence: about 1.6 s, never more than 8 s.
        XCTAssertGreaterThan(model.seenLengths[0], 16_000)
        XCTAssertLessThanOrEqual(model.seenLengths[0], 128_000)
        XCTAssertFalse(processor.isHoldingTurn)
    }

    func testVetoedPauseThenResumeIsOneSegment() {
        let vad = ScriptedVAD(script([(0.9, 1.0), (0.0, 0.4), (0.9, 1.0), (0.0, 0.5)]))
        let model = ScriptedTurnModel([0.2, 0.9])
        let processor = StreamingVADProcessor(
            provider: vad, config: vadConfig, turnCompletion: model,
            turnCompletionConfig: TurnCompletionConfig(threshold: 0.5, maxSilenceDuration: 1.0))
        let events = drive(processor, chunkCount: vad.probabilities.count)
        XCTAssertEqual(starts(events).count, 1, "resuming a held turn must not announce a new segment")
        let ended = segments(events)
        XCTAssertEqual(ended.count, 1)
        XCTAssertEqual(model.seenLengths.count, 2)
        XCTAssertGreaterThan(model.seenLengths[1], model.seenLengths[0])
        // The segment spans both bursts and the pause between them.
        XCTAssertEqual(ended[0].startTime, 0, accuracy: chunkSeconds)
        XCTAssertGreaterThan(ended[0].endTime, 2.3)
        XCTAssertFalse(processor.isHoldingTurn)
    }

    func testVetoedPauseEndsOnSilenceCap() {
        let vad = ScriptedVAD(script([(0.9, 1.0), (0.0, 0.3)]))
        let model = ScriptedTurnModel([0.1])
        let processor = StreamingVADProcessor(
            provider: vad, config: vadConfig, turnCompletion: model,
            turnCompletionConfig: TurnCompletionConfig(threshold: 0.5, maxSilenceDuration: 1.0))
        var events = drive(processor, chunkCount: vad.probabilities.count)
        XCTAssertTrue(segments(events).isEmpty)
        XCTAssertTrue(processor.isHoldingTurn)

        vad.probabilities.append(contentsOf: [Float](repeating: 0, count: chunks(1.0)))
        events += drive(processor, chunkCount: chunks(1.0))
        let ended = segments(events)
        XCTAssertEqual(ended.count, 1)
        XCTAssertEqual(model.seenLengths.count, 1, "the cap must not re-run the classifier")
        // The segment ends where the speech stopped, not when the cap expired.
        XCTAssertEqual(ended[0].endTime, 1.0, accuracy: 2 * chunkSeconds)
        XCTAssertFalse(processor.isHoldingTurn)
    }

    func testBriefBlipInsideHoldDoesNotResume() {
        // A single loud chunk during the hold falls below minSpeechDuration.
        let vad = ScriptedVAD(script([(0.9, 1.0), (0.0, 0.3), (0.9, 0.032), (0.0, 1.2)]))
        let model = ScriptedTurnModel([0.1])
        let processor = StreamingVADProcessor(
            provider: vad, config: vadConfig, turnCompletion: model,
            turnCompletionConfig: TurnCompletionConfig(threshold: 0.5, maxSilenceDuration: 1.0))
        let events = drive(processor, chunkCount: vad.probabilities.count)
        XCTAssertEqual(segments(events).count, 1)
        XCTAssertEqual(model.seenLengths.count, 1)
    }

    func testFlushSettlesHeldTurn() {
        let vad = ScriptedVAD(script([(0.9, 1.0), (0.0, 0.3)]))
        let model = ScriptedTurnModel([0.1])
        let processor = StreamingVADProcessor(
            provider: vad, config: vadConfig, turnCompletion: model)
        var events = drive(processor, chunkCount: vad.probabilities.count)
        XCTAssertTrue(processor.isHoldingTurn)
        events += processor.flush()
        let ended = segments(events)
        XCTAssertEqual(ended.count, 1)
        XCTAssertEqual(ended[0].endTime, 1.0, accuracy: 2 * chunkSeconds)
        XCTAssertFalse(processor.isHoldingTurn)
    }

    func testClassifierFailureFailsOpen() {
        let vad = ScriptedVAD(script([(0.9, 1.0), (0.0, 0.5)]))
        let model = ScriptedTurnModel([0.1])
        model.shouldThrow = true
        let processor = StreamingVADProcessor(
            provider: vad, config: vadConfig, turnCompletion: model)
        let events = drive(processor, chunkCount: vad.probabilities.count)
        XCTAssertEqual(segments(events).count, 1, "a failing classifier must not stall the turn")
        XCTAssertNil(processor.lastTurnCompletionProbability)
    }

    func testResetClearsHold() {
        let vad = ScriptedVAD(script([(0.9, 1.0), (0.0, 0.3)]))
        let model = ScriptedTurnModel([0.1])
        let processor = StreamingVADProcessor(
            provider: vad, config: vadConfig, turnCompletion: model)
        _ = drive(processor, chunkCount: vad.probabilities.count)
        XCTAssertTrue(processor.isHoldingTurn)
        processor.reset()
        XCTAssertFalse(processor.isHoldingTurn)
        XCTAssertNil(processor.lastTurnCompletionProbability)
    }
}

// MARK: - End-to-end

#if canImport(CoreML)
final class E2ESmartTurnTests: XCTestCase {
    private func loadModel() async throws -> SmartTurnModel {
        if let directory = ProcessInfo.processInfo.environment["SMART_TURN_COREML_MODEL_DIR"] {
            return try await SmartTurnModel.fromPretrained(
                cacheDir: URL(fileURLWithPath: directory, isDirectory: true),
                offlineMode: true)
        }
        return try await SmartTurnModel.fromPretrained()
    }

    func testE2EFinishedSentenceIsComplete() async throws {
        let model = try await loadModel()
        let audioURL = URL(fileURLWithPath: "Tests/Qwen3ASRTests/Resources/test_audio.wav")
        let (samples, sampleRate) = try AudioFileLoader.loadWAV(url: audioURL)
        // The fixture speaks from 5 s to 9 s and is silent afterwards; keep the
        // sentence plus three seconds of pause, like a VAD hand-off.
        let turn = Array(samples.prefix(12 * sampleRate))

        try model.prewarm()
        let first = try model.turnCompleteProbability(audio: turn, sampleRate: sampleRate)
        let second = try model.turnCompleteProbability(audio: turn, sampleRate: sampleRate)

        XCTAssertGreaterThan(first, 0.85, "reference (fp32 ONNX) is 0.97")
        XCTAssertEqual(first, second, accuracy: 1e-4)
    }

    func testE2ESyntheticInputsStayInRange() async throws {
        let model = try await loadModel()
        var tone = [Float](repeating: 0, count: 48_000)
        for index in tone.indices {
            let time = Float(index) / 16_000
            tone[index] = 0.2 * sin(2 * .pi * 140 * time) + 0.1 * sin(2 * .pi * 280 * time)
        }
        let probability = try model.turnCompleteProbability(audio: tone, sampleRate: 16_000)
        XCTAssertTrue(probability.isFinite)
        XCTAssertGreaterThanOrEqual(probability, 0)
        XCTAssertLessThanOrEqual(probability, 1)

        let silence = [Float](repeating: 0, count: 16_000)
        XCTAssertTrue(try model.turnCompleteProbability(audio: silence, sampleRate: 16_000).isFinite)

        // 48 kHz input is resampled and must land close to the 16 kHz result.
        var tone48 = [Float](repeating: 0, count: 144_000)
        for index in tone48.indices {
            let time = Float(index) / 48_000
            tone48[index] = 0.2 * sin(2 * .pi * 140 * time) + 0.1 * sin(2 * .pi * 280 * time)
        }
        let resampled = try model.turnCompleteProbability(audio: tone48, sampleRate: 48_000)
        XCTAssertEqual(resampled, probability, accuracy: 0.1)
    }
}
#endif
