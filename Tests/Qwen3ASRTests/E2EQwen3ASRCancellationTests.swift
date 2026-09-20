import XCTest
import Foundation
import MLX
@testable import Qwen3ASR
@testable import AudioCommon

/// Real-model contract checks for `transcribeCheckingCancellation`:
/// identical output to the synchronous API when nothing is cancelled, a
/// `CancellationError` when the calling task is cancelled mid-utterance,
/// and an unchanged synchronous API that still transcribes inside a
/// cancelled task.
final class E2EQwen3ASRCancellationTests: XCTestCase {
    static let modelId = "aufklarer/Qwen3-ASR-0.6B-MLX-4bit"
    static let targetSampleRate = 24000

    private func loadAudio() throws -> [Float] {
        guard let wavURL = Bundle.module.url(forResource: "test_audio", withExtension: "wav") else {
            throw XCTSkip("Test WAV file not found in bundle resources")
        }
        let (samples, sampleRate) = try AudioFileLoader.loadWAV(url: wavURL)
        if sampleRate == Self.targetSampleRate { return samples }
        return AudioFileLoader.resample(samples, from: sampleRate, to: Self.targetSampleRate)
    }

    func testUncancelledEntryPointMatchesSynchronousAPI() async throws {
        let model = try await Qwen3ASRModel.fromPretrained(modelId: Self.modelId)
        let audio = try loadAudio()
        let sync = model.transcribe(audio: audio, sampleRate: Self.targetSampleRate)
        let aware = try model.transcribeCheckingCancellation(
            audio: audio, sampleRate: Self.targetSampleRate)
        XCTAssertFalse(sync.isEmpty, "Transcription should not be empty")
        XCTAssertEqual(aware, sync, "Cancellation-aware decode must match the synchronous API")
    }

    func testCancelledTaskThrowsWhileSynchronousAPIStillTranscribes() async throws {
        let model = try await Qwen3ASRModel.fromPretrained(modelId: Self.modelId)
        let audio = try loadAudio()

        // 20 s of audio takes far longer than the delay below to encode and
        // decode, so the cancel lands while work is still in flight.
        let task = Task.detached { () throws -> String in
            try model.transcribeCheckingCancellation(
                audio: audio, sampleRate: Self.targetSampleRate)
        }
        try await Task.sleep(for: .milliseconds(50))
        task.cancel()
        switch await task.result {
        case .success(let text):
            XCTFail("Expected CancellationError, got transcript: \"\(text)\"")
        case .failure(let error):
            XCTAssertTrue(error is CancellationError, "Expected CancellationError, received \(error)")
        }

        let text = await Task.detached { () -> String in
            withUnsafeCurrentTask { $0?.cancel() }
            return model.transcribe(audio: audio, sampleRate: Self.targetSampleRate)
        }.value
        XCTAssertFalse(text.isEmpty, "Synchronous transcribe must not observe task cancellation")
    }
}
