#if canImport(AVFoundation)
import AVFoundation
import XCTest
@testable import AudioCommon

final class AudioFileStreamTests: XCTestCase {
    func testStreamMixesAllChannelsAndMarksOnlyLastChunkFinal() async throws {
        let url = try makeStereoFile(
            sampleRate: 16_000,
            frames: 1_000,
            left: 0.8,
            right: 0.2)
        defer { try? FileManager.default.removeItem(at: url) }

        let source = AudioFileLoader.stream(
            url: url,
            options: AudioFileStreamOptions(
                targetSampleRate: 16_000,
                chunkDuration: 0.02))
        var chunks: [CapturedAudioChunk] = []
        for try await chunk in source { chunks.append(chunk) }

        XCTAssertEqual(chunks.map(\.samples.count), [320, 320, 320, 40])
        XCTAssertEqual(chunks.map(\.frameIndex), [0, 320, 640, 960])
        XCTAssertEqual(chunks.map(\.isFinal), [false, false, false, true])
        XCTAssertEqual(chunks.flatMap(\.samples).count, 1_000)
        XCTAssertTrue(chunks.flatMap(\.samples).allSatisfy {
            abs($0 - 0.5) < 0.0001
        })
    }

    func testSelectedChannelDoesNotDiscardSpeechOutsideChannelZero() throws {
        let url = try makeStereoFile(
            sampleRate: 16_000,
            frames: 256,
            left: 0,
            right: 0.75)
        defer { try? FileManager.default.removeItem(at: url) }

        let selected = try AudioFileLoader.load(
            url: url,
            targetSampleRate: 16_000,
            channelSelection: .select([1]))
        let first = try AudioFileLoader.load(
            url: url,
            targetSampleRate: 16_000,
            channelSelection: .first)
        let mixed = try AudioFileLoader.load(
            url: url,
            targetSampleRate: 16_000)

        XCTAssertTrue(selected.allSatisfy { abs($0 - 0.75) < 0.0001 })
        XCTAssertTrue(first.allSatisfy { abs($0) < 0.0001 })
        XCTAssertTrue(mixed.allSatisfy { abs($0 - 0.375) < 0.0001 })
    }

    func testInvalidChannelSelectionFailsBeforeReading() throws {
        let url = try makeStereoFile(
            sampleRate: 16_000,
            frames: 32,
            left: 0,
            right: 0)
        defer { try? FileManager.default.removeItem(at: url) }

        XCTAssertThrowsError(try AudioFileChunkReader(
            url: url,
            options: AudioFileStreamOptions(
                channelSelection: .select([2])))) { error in
            guard case AudioLoadError.invalidChannelSelection(
                let requested, let available) = error else {
                return XCTFail("Unexpected error: \(error)")
            }
            XCTAssertEqual(requested, [2])
            XCTAssertEqual(available, 2)
        }
    }

    func testStreamingResamplerDrainsAndPreservesContinuousOffsets() throws {
        let url = try makeStereoFile(
            sampleRate: 48_000,
            frames: 4_800,
            left: 0.4,
            right: 0.4)
        defer { try? FileManager.default.removeItem(at: url) }

        let reader = try AudioFileChunkReader(
            url: url,
            options: AudioFileStreamOptions(
                targetSampleRate: 16_000,
                chunkDuration: 0.03))
        var chunks: [CapturedAudioChunk] = []
        while let chunk = try reader.readChunk() { chunks.append(chunk) }

        let samples = chunks.flatMap(\.samples)
        XCTAssertEqual(samples.count, 1_600)
        XCTAssertEqual(chunks.last?.isFinal, true)
        for index in chunks.indices.dropFirst() {
            XCTAssertEqual(
                chunks[index].frameIndex,
                chunks[index - 1].frameIndex
                    + Int64(chunks[index - 1].samples.count))
        }
    }

    private func makeStereoFile(
        sampleRate: Double,
        frames: Int,
        left: Float,
        right: Float
    ) throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("caf")
        guard let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: sampleRate,
            channels: 2,
            interleaved: false),
              let buffer = AVAudioPCMBuffer(
                pcmFormat: format,
                frameCapacity: AVAudioFrameCount(frames)) else {
            throw AudioLoadError.bufferCreationFailed
        }
        buffer.frameLength = AVAudioFrameCount(frames)
        for frame in 0..<frames {
            buffer.floatChannelData![0][frame] = left
            buffer.floatChannelData![1][frame] = right
        }
        let file = try AVAudioFile(forWriting: url, settings: format.settings)
        try file.write(from: buffer)
        return url
    }
}
#endif
