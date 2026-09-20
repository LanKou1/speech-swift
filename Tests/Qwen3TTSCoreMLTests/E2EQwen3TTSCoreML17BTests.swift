import XCTest
import CoreML
import Foundation
import AudioCommon
@testable import Qwen3TTSCoreML
@testable import Qwen3ASR

/// Opt in with QWEN3TTS_17B_BUNDLE and a prepared 2048-channel speaker embedding.
/// The public bundle intentionally has no default voice. A configured path that
/// fails to load is a test failure, not a skip.
final class E2EQwen3TTSCoreML17BTests: XCTestCase {
    private func loadModel() async throws -> Qwen3TTSCoreMLModel {
        guard let path = ProcessInfo.processInfo.environment["QWEN3TTS_17B_BUNDLE"] else {
            throw XCTSkip("Set QWEN3TTS_17B_BUNDLE to the 1.7B compiled bundle")
        }
        let speaker = ProcessInfo.processInfo.environment["QWEN3TTS_17B_SPEAKER"]
            ?? URL(fileURLWithPath: path).appendingPathComponent("speaker_embedding.npy").path
        return try await Qwen3TTSCoreMLModel.fromPretrained(
            localPath: path, computeUnits: .cpuOnly,
            speakerEmbeddingURL: URL(fileURLWithPath: speaker))
    }

    func testRoundTripSynthesizeTranscribe() async throws {
        let model = try await loadModel()
        defer { model.unload() }
        XCTAssertEqual(model.hiddenSize, 2048)
        XCTAssertEqual(model.maxSequenceLength, 1024)
        XCTAssertEqual(model.maximumAudioFrames, 125)
        let text = "The quick brown fox jumps over the lazy dog."
        let start = Date()
        let audio = try model.synthesize(text: text, temperature: 0, topK: 1)
        let elapsed = Date().timeIntervalSince(start)
        XCTAssertGreaterThan(audio.count, 2400)
        XCTAssertTrue(audio.allSatisfy(\.isFinite))
        model.unload()
        if let directory = ProcessInfo.processInfo.environment["QWEN3TTS_E2E_ARTIFACTS"] {
            try WAVWriter.write(samples: audio, sampleRate: 24000,
                                to: URL(fileURLWithPath: directory).appendingPathComponent("swift-17b.wav"))
        }
        let asr = try await Qwen3ASRModel.fromPretrained()
        let transcript = asr.transcribe(audio: audio, sampleRate: 24000)
        let keywords = ["quick", "brown", "fox", "jumps", "lazy", "dog"]
        let matched = keywords.filter { transcript.lowercased().contains($0) }
        print("[1.7B ROUNDTRIP] \(transcript); \(matched.count)/6 keywords; wall=\(elapsed)s audio=\(Double(audio.count)/24000)s")
        XCTAssertGreaterThanOrEqual(matched.count, 5, transcript)
    }

    func testPromptBeyond256PositionsAndFreshStateReset() async throws {
        let model = try await loadModel()
        defer { model.unload() }
        // Each repeated word is at least one token, exceeding the former 256-position cap.
        let text = Array(repeating: "Hello", count: 270).joined(separator: " ")
        let first = try model.synthesize(text: text, temperature: 0, topK: 1, maxTokens: 2)
        let second = try model.synthesize(text: text, temperature: 0, topK: 1, maxTokens: 2)
        XCTAssertEqual(first.count, 2 * 1920)
        XCTAssertTrue(first.allSatisfy(\.isFinite))
        XCTAssertEqual(first, second, "Each request must reset its talker cache")
    }

    func testRejectsFrameOverflowAndWrongSpeakerWidth() async throws {
        let model = try await loadModel()
        defer { model.unload() }
        XCTAssertThrowsError(try model.synthesize(text: "Hello", maxTokens: 126))
        XCTAssertThrowsError(try model.synthesize(text: "Hello", maxTokens: 0))
        model.speakerEmbedding = try MLMultiArray(shape: [1, 1024, 1, 1], dataType: .float32)
        XCTAssertThrowsError(try model.synthesize(text: "Hello"))
        model.speakerEmbedding = nil
        XCTAssertThrowsError(try model.synthesize(text: "Hello"))
    }
}
