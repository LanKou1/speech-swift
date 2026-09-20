import XCTest
import CoreML
@testable import Qwen3TTSCoreML

final class BundleConfigurationTests: XCTestCase {
    private func config(_ json: String) throws -> BundleConfiguration {
        try JSONDecoder().decode(BundleConfiguration.self, from: Data(json.utf8))
    }

    func testLegacyBundleDefaults() throws {
        let c = try config(#"{"hidden_size":1024,"max_seq_len":256,"max_codec_tokens":125}"#)
        XCTAssertEqual(c.hiddenSize, 1024)
        XCTAssertEqual(c.maxSequenceLength, 256)
        XCTAssertEqual(c.speechDecoderFrames, 125)
        XCTAssertEqual(c.predictorKVDimension, 5120)
        XCTAssertEqual(try c.generationLimit(requested: 125, promptCount: 20), 125)
    }

    func testLargeBundleDoesNotConfuseTalkerAndPredictorWidths() throws {
        let c = try config(#"{"hidden_size":2048,"code_predictor_hidden_size":1024,"max_seq_len":1024,"speech_decoder_frames":125,"predictor_kv_dim":5120,"requires_speaker_embedding":true}"#)
        XCTAssertEqual(c.hiddenSize, 2048)
        XCTAssertEqual(c.predictorKVDimension, 5120)
        XCTAssertTrue(c.requiresSpeakerEmbedding)
        XCTAssertEqual(try c.generationLimit(requested: 125, promptCount: 300), 125)
        XCTAssertEqual(try c.generationLimit(requested: 125, promptCount: 1023), 1)
        XCTAssertThrowsError(try c.generationLimit(requested: 125, promptCount: 1024))
    }

    func testRejectInvalidConfiguration() throws {
        for json in [#"{"hidden_size":1025}"#, #"{"max_seq_len":0}"#,
                     #"{"speech_decoder_frames":-1}"#, #"{"predictor_kv_dim":0}"#,
                     #"{"sample_rate":48000}"#, #"{"samples_per_frame":1000}"#] {
            XCTAssertThrowsError(try config(json), json)
        }
    }

    func testGenerationBoundsRejectOverflowBeforeInference() throws {
        let c = try config("{}")
        for limit in [-1, 0, 126] { XCTAssertThrowsError(try c.generationLimit(requested: limit, promptCount: 20)) }
        for count in [0, 256, 300] { XCTAssertThrowsError(try c.generationLimit(requested: 125, promptCount: count)) }
        XCTAssertEqual(try c.generationLimit(requested: 1, promptCount: 20), 1)
    }

    private func npy(_ values: [Float], dtype: String = "<f4", shape: String? = nil, version: UInt8 = 1) -> Data {
        let header = "{'descr': '\(dtype)', 'fortran_order': False, 'shape': (\(shape ?? "\(values.count),")), }\n"
        var data = Data([0x93, 78, 85, 77, 80, 89, version, 0])
        let length = header.utf8.count
        data.append(contentsOf: [UInt8(length & 255), UInt8((length >> 8) & 255)])
        if version == 2 { data.append(contentsOf: [0, 0]) }
        data.append(Data(header.utf8))
        for value in values {
            var bits = value.bitPattern.littleEndian
            withUnsafeBytes(of: &bits) { data.append(contentsOf: $0) }
        }
        return data
    }

    func testEmbeddingFilesForBothModelWidths() throws {
        for width in [1024, 2048] {
            let values = (0..<width).map { Float($0) / 1000 }
            XCTAssertEqual(try EmbeddingFile.read(npy(values), channels: width), values)
            XCTAssertEqual(try EmbeddingFile.read(npy(values, shape: "1, \(width), 1, 1", version: 2), channels: width), values)
        }
    }

    func testEmbeddingRejectsWrongWidthDtypeAndNonfiniteValues() throws {
        let values = [Float](repeating: 1, count: 1024)
        XCTAssertThrowsError(try EmbeddingFile.read(npy(values), channels: 2048))
        XCTAssertThrowsError(try EmbeddingFile.read(npy(values, dtype: ">f4"), channels: 1024))
        XCTAssertThrowsError(try EmbeddingFile.read(npy(values, shape: "32, 32"), channels: 1024))
        var bad = values; bad[100] = .nan
        XCTAssertThrowsError(try EmbeddingFile.read(npy(bad), channels: 1024))
        let valid = npy(values)
        for count in [0, 5, 9, 11, valid.count - 1] {
            XCTAssertThrowsError(try EmbeddingFile.read(Data(valid.prefix(count)), channels: 1024))
        }
    }

    func testEmbeddingAdditionPreserves2048ChannelsAndStrides() throws {
        let width = 2048
        let pointer = UnsafeMutablePointer<Float>.allocate(capacity: width * 2)
        pointer.initialize(repeating: 0, count: width * 2)
        let strided = try MLMultiArray(dataPointer: pointer, shape: [1, NSNumber(value: width), 1, 1],
                                      dataType: .float32, strides: [NSNumber(value: width * 2), 2, 1, 1],
                                      deallocator: { $0.assumingMemoryBound(to: Float.self).deallocate() })
        let other = try MLMultiArray(shape: [1, NSNumber(value: width), 1, 1], dataType: .float16)
        for i in 0..<width { pointer[2*i] = Float(i) / 2048; other[[0, NSNumber(value: i), 0, 0]] = 1 }
        let result = addMLMultiArrays(strided, other)
        XCTAssertEqual(result.count, width)
        XCTAssertEqual(result[[0, 2047, 0, 0]].floatValue, Float(Float16(1 + Float(2047) / 2048)))
    }
    func testVocoderRejectsMalformedCodebooksAndOverflow() throws {
        let valid = Array(repeating: [Int32](repeating: 0, count: 125), count: 16)
        XCTAssertEqual(try SpeechDecoderCoreML.validate(codes: valid, capacity: 125), 125)
        XCTAssertThrowsError(try SpeechDecoderCoreML.validate(codes: [], capacity: 125))
        XCTAssertThrowsError(try SpeechDecoderCoreML.validate(codes: valid, capacity: 124))
        var uneven = valid; uneven[1].removeLast()
        XCTAssertThrowsError(try SpeechDecoderCoreML.validate(codes: uneven, capacity: 125))
        var badToken = valid; badToken[0][0] = 2048
        XCTAssertThrowsError(try SpeechDecoderCoreML.validate(codes: badToken, capacity: 125))
    }

}
