#if canImport(AVFoundation)
import AVFoundation
#endif
import XCTest
@testable import AudioCommon

final class CapturedAudioChunkTests: XCTestCase {
    func testChunkPreservesSamplesRateAndOptionalHostTime() {
        let chunk = CapturedAudioChunk(
            samples: [0.25, -0.5], sampleRate: 16_000, hostTime: 123)

        XCTAssertEqual(chunk.samples, [0.25, -0.5])
        XCTAssertEqual(chunk.sampleRate, 16_000)
        XCTAssertEqual(chunk.hostTime, 123)
        XCTAssertEqual(chunk.frameIndex, 0)
        XCTAssertFalse(chunk.isFinal)
    }

    func testStreamingRecognitionUpdateCarriesUnifiedMetadata() {
        let word = TimedWord(text: "hello", startTime: 0.1, endTime: 0.4)
        let update = StreamingRecognitionUpdate(
            text: "hello",
            isFinal: true,
            endOfUtterance: true,
            segmentIndex: 2,
            confidence: 0.9,
            language: "en",
            startTime: 0,
            endTime: 0.5,
            words: [word])

        XCTAssertEqual(update.text, "hello")
        XCTAssertTrue(update.isFinal)
        XCTAssertTrue(update.endOfUtterance)
        XCTAssertEqual(update.segmentIndex, 2)
        XCTAssertEqual(update.words, [word])
    }

    #if canImport(AVFoundation)
    func testAudioIOKeepsEchoCancellationAndPlaybackChoicesIndependent() {
        let captureOnly = AudioIO(enableAEC: true, enablePlayback: false)
        XCTAssertTrue(captureOnly.enableAEC)
        XCTAssertFalse(captureOnly.enablePlayback)

        let defaultIO = AudioIO()
        XCTAssertFalse(defaultIO.enableAEC)
        XCTAssertTrue(defaultIO.enablePlayback)
    }

    func testAudioIORequestsVoiceProcessingOnlyWhenAECIsEnabled() throws {
        var requests: [Bool] = []

        try AudioIO.enableVoiceProcessingIfRequested(false) {
            requests.append($0)
        }
        XCTAssertTrue(requests.isEmpty)

        try AudioIO.enableVoiceProcessingIfRequested(true) {
            requests.append($0)
        }
        XCTAssertEqual(requests, [true])
    }

    func testAudioIOPropagatesVoiceProcessingFailure() {
        XCTAssertThrowsError(
            try AudioIO.enableVoiceProcessingIfRequested(true) { _ in
                throw VoiceProcessingTestError.unavailable
            }
        ) { error in
            XCTAssertEqual(error as? VoiceProcessingTestError, .unavailable)
        }
    }

    func testAudioIOPreservesOnlyValidHostTime() {
        XCTAssertEqual(AudioIO.hostTime(from: AVAudioTime(hostTime: 42)), 42)
        XCTAssertNil(AudioIO.hostTime(from: AVAudioTime(sampleTime: 0, atRate: 16_000)))
    }
    #endif
}

private enum VoiceProcessingTestError: Error {
    case unavailable
}
