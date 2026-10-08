import Foundation
import XCTest
@testable import Qwen3TTS

/// Synthetic metadata only. No MLX allocation, model download, inference or audio.
final class WeightValidationTests: XCTestCase {
    func testCompleteF16DecoderEnablesHalfMath() throws {
        let (config, tensors) = decoderFixture(dtype: "F16")
        XCTAssertTrue(try TTSWeightLoader.validateDecoderWeights(tensors, config: config))
    }

    func testCompleteF32DecoderKeepsOriginalMath() throws {
        let (config, tensors) = decoderFixture(dtype: "F32")
        XCTAssertFalse(try TTSWeightLoader.validateDecoderWeights(tensors, config: config))
    }

    func testEveryRequiredDecoderTensorIsRequired() throws {
        let (config, tensors) = decoderFixture(dtype: "F16")
        for key in tensors.keys.sorted() {
            var incomplete = tensors
            incomplete.removeValue(forKey: key)
            XCTAssertThrowsError(try TTSWeightLoader.validateDecoderWeights(incomplete, config: config), key)
        }
    }

    func testEveryRequiredDecoderShapeIsChecked() throws {
        let (config, tensors) = decoderFixture(dtype: "F16")
        for (key, value) in tensors {
            var damaged = tensors
            damaged[key] = .init(shape: value.shape + [2], dtype: value.dtype)
            XCTAssertThrowsError(try TTSWeightLoader.validateDecoderWeights(damaged, config: config), key)
        }
    }

    func testCompleteDecoderWithOneF32ConvolutionIsRejected() throws {
        let (config, tensors) = decoderFixture(dtype: "F16")
        var mixed = tensors
        let key = "decoder.pre_conv.conv.weight"
        mixed[key] = .init(shape: try XCTUnwrap(tensors[key]).shape, dtype: "F32")
        XCTAssertThrowsError(try TTSWeightLoader.validateDecoderWeights(mixed, config: config))
    }

    func testCompleteDecoderWithOnlyFinalF16ConvolutionIsRejected() throws {
        let (config, tensors) = decoderFixture(dtype: "F32")
        var mixed = tensors
        let key = "decoder.decoder.6.conv.weight"
        mixed[key] = .init(shape: try XCTUnwrap(tensors[key]).shape, dtype: "F16")
        XCTAssertThrowsError(try TTSWeightLoader.validateDecoderWeights(mixed, config: config))
    }

    func testCompletePrecomputedCodebooksAreAccepted() throws {
        let (config, tensors) = decoderFixture(dtype: "F16")
        var precomputed = tensors
        for name in ["rvq_first", "rvq_rest"] {
            let p = "decoder.quantizer.\(name).vq.layers.0._codebook"
            precomputed.removeValue(forKey: p + ".cluster_usage")
            precomputed.removeValue(forKey: p + ".embedding_sum")
            precomputed[p + ".embed"] = .init(shape: [8, 4], dtype: "F16")
        }
        XCTAssertTrue(try TTSWeightLoader.validateDecoderWeights(precomputed, config: config))
    }

    func testValidSafetensorsHeaderAndExtent() throws {
        try withDirectory { directory in
            try writeTensor(to: directory.appendingPathComponent("model.safetensors"))
            let headers = try TTSWeightLoader.readWeightHeaders(from: directory)
            XCTAssertEqual(headers["fixture"]?.shape, [2])
            XCTAssertEqual(headers["fixture"]?.dtype, "F32")
        }
    }

    func testTruncatedSafetensorsPayloadIsRejected() throws {
        try withDirectory { directory in
            try writeTensor(to: directory.appendingPathComponent("model.safetensors"), payloadBytes: 4)
            XCTAssertThrowsError(try TTSWeightLoader.readWeightHeaders(from: directory))
        }
    }

    func testIncorrectSafetensorsShapeByteCountIsRejected() throws {
        try withDirectory { directory in
            try writeTensor(to: directory.appendingPathComponent("model.safetensors"), shape: [3])
            XCTAssertThrowsError(try TTSWeightLoader.readWeightHeaders(from: directory))
        }
    }

    func testDuplicateShardTensorIsRejected() throws {
        try withDirectory { directory in
            try writeTensor(to: directory.appendingPathComponent("a.safetensors"))
            try writeTensor(to: directory.appendingPathComponent("b.safetensors"))
            XCTAssertThrowsError(try TTSWeightLoader.readWeightHeaders(from: directory))
        }
    }

    func testConfigOnlyBundleIsRejected() throws {
        try withDirectory { directory in
            try Data(#"{"model_size":"0.6b"}"#.utf8).write(to: directory.appendingPathComponent("config.json"))
            XCTAssertThrowsError(try TTSWeightLoader.validateCloneWeightSet(
                modelDirectory: directory, codecDirectory: directory))
        }
    }

    func testQuantizedEmbeddingCanUseDifferentBitWidthFromLinears() throws {
        let (talker, predictor, tensors) = talkerFixture()
        XCTAssertNoThrow(try TTSWeightLoader.validateTalkerWeights(tensors, talker: talker, predictor: predictor))
    }

    func testPartialQuantizedTripletIsRejected() throws {
        let (talker, predictor, tensors) = talkerFixture()
        for suffix in ["weight", "scales", "biases"] {
            var incomplete = tensors
            incomplete.removeValue(forKey: "talker.text_projection.linear_fc1." + suffix)
            XCTAssertThrowsError(try TTSWeightLoader.validateTalkerWeights(incomplete, talker: talker, predictor: predictor))
        }
    }

    func testWrongPackedBitWidthIsRejected() throws {
        let (talker, predictor, tensors) = talkerFixture()
        var wrong = tensors
        wrong["talker.text_projection.linear_fc1.weight"] = .init(shape: [64, 16], dtype: "U32")
        XCTAssertThrowsError(try TTSWeightLoader.validateTalkerWeights(wrong, talker: talker, predictor: predictor))
    }

    func testEveryTalkerTensorIsRequiredAndShapeChecked() throws {
        let (talker, predictor, tensors) = talkerFixture()
        for (key, value) in tensors {
            var incomplete = tensors
            incomplete.removeValue(forKey: key)
            XCTAssertThrowsError(try TTSWeightLoader.validateTalkerWeights(incomplete, talker: talker, predictor: predictor), key)
            var wrong = tensors
            wrong[key] = .init(shape: value.shape + [2], dtype: value.dtype)
            XCTAssertThrowsError(try TTSWeightLoader.validateTalkerWeights(wrong, talker: talker, predictor: predictor), key)
        }
    }

    func testUnsafePositiveModelDimensionsAreRejectedBeforeSchemaLoops() throws {
        for fields in [["num_hidden_layers": Int.max], ["head_dim": Int.max],
                       ["num_attention_heads": Int.max]] {
            try withDirectory { directory in
                let object: [String: Any] = ["model_size": "0.6b", "talker_config": fields]
                try JSONSerialization.data(withJSONObject: object).write(to: directory.appendingPathComponent("config.json"))
                XCTAssertThrowsError(try TTSWeightLoader.validateCloneWeightSet(
                    modelDirectory: directory, codecDirectory: directory)) { error in
                    XCTAssertEqual((error as? TTSWeightValidationError)?.reason, "unsupported or unsafe model dimensions")
                }
            }
        }
    }

    func testZeroAndNegativeModelDimensionsAreRejectedBeforeAllocation() throws {
        for fields in [["num_hidden_layers": -1], ["head_dim": 0], ["num_key_value_heads": 0]] {
            try withDirectory { directory in
                let object: [String: Any] = ["model_size": "0.6b", "talker_config": fields]
                try JSONSerialization.data(withJSONObject: object).write(to: directory.appendingPathComponent("config.json"))
                XCTAssertThrowsError(try TTSWeightLoader.validateCloneWeightSet(
                    modelDirectory: directory, codecDirectory: directory)) { error in
                    XCTAssertEqual((error as? TTSWeightValidationError)?.reason, "model configuration cannot be resolved")
                }
            }
        }
    }

    func testInvalidQuantizationGroupIsRejectedBeforeAllocation() throws {
        for group in [-1, 0, Int.max] {
            try withDirectory { directory in
                let object: [String: Any] = ["model_size": "0.6b", "quantization_config": ["bits": 4, "group_size": group]]
                try JSONSerialization.data(withJSONObject: object).write(to: directory.appendingPathComponent("config.json"))
                XCTAssertThrowsError(try TTSWeightLoader.validateCloneWeightSet(
                    modelDirectory: directory, codecDirectory: directory)) { error in
                    XCTAssertTrue(error is TTSWeightValidationError)
                }
            }
        }
    }

    func testWrongModelFamilyIsRejected() throws {
        try withDirectory { directory in
            try Data(#"{"model_size":"0.6b","model_type":"wrong-family"}"#.utf8)
                .write(to: directory.appendingPathComponent("config.json"))
            XCTAssertThrowsError(try TTSWeightLoader.validateCloneWeightSet(
                modelDirectory: directory, codecDirectory: directory)) { error in
                XCTAssertEqual((error as? TTSWeightValidationError)?.reason, "model configuration cannot be resolved")
            }
        }
    }

    func testMissingCorruptAndWrongCodecConfigurationsAreRejected() throws {
        try withDirectory { directory in
            let model = directory.appendingPathComponent("model")
            let codec = directory.appendingPathComponent("codec")
            try FileManager.default.createDirectory(at: model, withIntermediateDirectories: true)
            try FileManager.default.createDirectory(at: codec, withIntermediateDirectories: true)
            try Data(#"{"model_size":"0.6b"}"#.utf8).write(to: model.appendingPathComponent("config.json"))
            for contents in [nil, "{broken", #"{"model_type":"wrong-family","decoder_config":{},"encoder_config":{}}"#] as [String?] {
                if let contents { try Data(contents.utf8).write(to: codec.appendingPathComponent("config.json")) }
                XCTAssertThrowsError(try TTSWeightLoader.validateCloneWeightSet(modelDirectory: model, codecDirectory: codec)) { error in
                    XCTAssertEqual((error as? TTSWeightValidationError)?.reason, "codec configuration cannot be resolved")
                }
            }
        }
    }

    private func withDirectory(_ body: (URL) throws -> Void) throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("weight-validation-\(UUID())")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try body(directory)
    }

    private func writeTensor(to url: URL, shape: [Int] = [2], payloadBytes: Int = 8) throws {
        let header = try JSONSerialization.data(withJSONObject: [
            "fixture": ["dtype": "F32", "shape": shape, "data_offsets": [0, 8]]
        ])
        var count = UInt64(header.count).littleEndian
        var file = withUnsafeBytes(of: &count) { Data($0) }
        file.append(header)
        file.append(Data(repeating: 0, count: payloadBytes))
        try file.write(to: url)
    }

    private func talkerFixture() -> (TalkerConfig, CodePredictorConfig, TTSWeightHeaders) {
        var t = TalkerConfig()
        t.hiddenSize = 64; t.textHiddenSize = 64; t.textVocabSize = 8; t.codecVocabSize = 8
        t.numLayers = 0; t.bits = 4; t.groupSize = 64
        var c = CodePredictorConfig()
        c.hiddenSize = 64; c.embeddingDim = 64; c.vocabSize = 8; c.numCodeGroups = 2
        c.numLayers = 0; c.bits = 4; c.groupSize = 64
        var w: TTSWeightHeaders = [:]
        func add(_ key: String, _ shape: [Int], _ dtype: String = "BF16") { w[key] = .init(shape: shape, dtype: dtype) }
        func quant(_ key: String, _ input: Int, _ output: Int, _ bits: Int = 4, bias: Bool = false) {
            add(key + ".weight", [output, input * bits / 32], "U32")
            add(key + ".scales", [output, input / 64]); add(key + ".biases", [output, input / 64])
            if bias { add(key + ".bias", [output]) }
        }
        add("talker.model.codec_embedding.weight", [8, 64])
        quant("talker.model.text_embedding", 64, 8, 8)
        quant("talker.text_projection.linear_fc1", 64, 64, bias: true)
        quant("talker.text_projection.linear_fc2", 64, 64, bias: true)
        quant("talker.codec_head", 64, 8)
        add("talker.model.norm.weight", [64])
        add("talker.code_predictor.model.codec_embedding.0.weight", [8, 64])
        add("talker.code_predictor.model.norm.weight", [64])
        quant("talker.code_predictor.lm_head.0", 64, 8)
        return (t, c, w)
    }

    private func decoderFixture(dtype: String) -> (SpeechTokenizerDecoderConfig, TTSWeightHeaders) {
        var c = SpeechTokenizerDecoderConfig()
        c.hiddenSize = 8; c.latentDim = 8; c.decoderDim = 64; c.codebookDim = 4
        c.numLayers = 1; c.numHeads = 2; c.headDim = 4; c.numQuantizers = 2
        c.semanticCodebookSize = 8; c.acousticCodebookSize = 8
        var w: TTSWeightHeaders = [:]
        func add(_ key: String, _ shape: [Int]) { w[key] = .init(shape: shape, dtype: dtype) }
        func conv(_ key: String, _ shape: [Int], _ bias: Int?) {
            add(key + ".weight", shape)
            if let bias { add(key + ".bias", [bias]) }
        }
        func snake(_ key: String, _ dim: Int) { add(key + ".alpha", [dim]); add(key + ".beta", [dim]) }
        for name in ["rvq_first", "rvq_rest"] {
            let p = "decoder.quantizer.\(name)"
            add(p + ".output_proj.weight", [8, 4, 1])
            add(p + ".vq.layers.0._codebook.cluster_usage", [8])
            add(p + ".vq.layers.0._codebook.embedding_sum", [8, 4])
        }
        conv("decoder.pre_conv.conv", [8, 8, 3], 8)
        conv("decoder.pre_transformer.input_proj", [8, 8], 8)
        conv("decoder.pre_transformer.output_proj", [8, 8], 8)
        add("decoder.pre_transformer.norm.weight", [8])
        let l = "decoder.pre_transformer.layers.0"
        for p in ["q_proj", "k_proj", "v_proj", "o_proj"] { add(l + ".self_attn." + p + ".weight", [8, 8]) }
        for p in ["input_layernorm", "post_attention_layernorm"] { add(l + "." + p + ".weight", [8]) }
        for p in ["gate_proj", "up_proj"] { add(l + ".mlp." + p + ".weight", [16, 8]) }
        add(l + ".mlp.down_proj.weight", [8, 16])
        for p in ["self_attn_layer_scale", "mlp_layer_scale"] { add(l + "." + p + ".scale", [8]) }
        for i in 0..<2 {
            let p = "decoder.upsample.\(i)"
            conv(p + ".0.conv", [8, 8, 2], 8)
            conv(p + ".1.dwconv.conv", [8, 1, 7], 8)
            add(p + ".1.norm.weight", [8]); add(p + ".1.norm.bias", [8]); add(p + ".1.gamma", [8])
            conv(p + ".1.pwconv1", [32, 8], 32); conv(p + ".1.pwconv2", [8, 32], 8)
        }
        conv("decoder.decoder.0.conv", [64, 8, 7], 64)
        var dim = 64
        for (i, rate) in [8, 5, 4, 3].enumerated() {
            let p = "decoder.decoder.\(i + 1).block"
            snake(p + ".0", dim)
            conv(p + ".1.conv", [dim, dim / 2, rate * 2], dim / 2)
            dim /= 2
            for j in 2...4 {
                snake(p + ".\(j).act1", dim); snake(p + ".\(j).act2", dim)
                conv(p + ".\(j).conv1.conv", [dim, dim, 7], dim)
                conv(p + ".\(j).conv2.conv", [dim, dim, 1], dim)
            }
        }
        snake("decoder.decoder.5", 4)
        conv("decoder.decoder.6.conv", [1, 4, 7], 1)
        return (c, w)
    }
}
