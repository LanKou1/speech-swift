import Foundation
import CoreFoundation

/// A local weight bundle did not match the architecture it would be loaded into.
/// Reasons contain parameter names and format information, never tensor contents.
public struct TTSWeightValidationError: Error, LocalizedError {
    public let reason: String
    public var errorDescription: String? { "Invalid TTS weight bundle: \(reason)" }
}

/// Header-only description: validation does not allocate an MLX model or map its weights.
struct TTSWeightTensor {
    let shape: [Int]
    let dtype: String
}

typealias TTSWeightHeaders = [String: TTSWeightTensor]

extension TTSWeightLoader {
    /// Validate both halves of a local clone bundle before selecting or allocating it.
    /// Covers every parameter consumed by the talker, code predictor, speaker encoder,
    /// speech decoder and reference encoder. Safetensors structure and payload extents
    /// are checked too. This is structural validation, not a cryptographic checksum.
    public static func validateCloneWeightSet(modelDirectory: URL, codecDirectory: URL) throws {
        guard modelDirectory.isFileURL, codecDirectory.isFileURL else {
            throw invalidWeights("local file URLs are required")
        }
        let config: Qwen3TTSConfig
        do {
            let configURL = modelDirectory.appendingPathComponent("config.json")
            let metadata = try JSONSerialization.jsonObject(with: Data(contentsOf: configURL)) as? [String: Any]
            guard let metadata,
                  metadata["model_type"] == nil || metadata["model_type"] as? String == "qwen3_tts",
                  metadata["architectures"] == nil || metadata["architectures"] as? [String] == ["Qwen3TTSForConditionalGeneration"] else {
                throw invalidWeights("unsupported model family")
            }
            config = try Qwen3TTSModel.resolveLocalConfiguration(
                from: modelDirectory.appendingPathComponent("config.json"))
        } catch {
            throw invalidWeights("model configuration cannot be resolved")
        }
        try validateConfigurationBounds(config)
        try validateCodecConfiguration(at: codecDirectory.appendingPathComponent("config.json"))
        let model = try readWeightHeaders(from: modelDirectory)
        try validateTalkerWeights(model, talker: config.talker, predictor: config.codePredictor)
        try validateSpeakerWeights(model, embeddingDim: config.talker.hiddenSize)
        let codec = try readWeightHeaders(from: codecDirectory)
        _ = try validateDecoderWeights(codec, config: config.speechTokenizerDecoder)
        try validateEncoderWeights(codec)
    }

    private static func validateConfigurationBounds(_ config: Qwen3TTSConfig) throws {
        let t = config.talker, p = config.codePredictor
        let dimensions = [t.hiddenSize, t.numHeads, t.numKVHeads, t.headDim, t.intermediateSize,
                          t.textVocabSize, t.textHiddenSize, t.codecVocabSize, p.hiddenSize,
                          p.embeddingDim, p.numHeads, p.numKVHeads, p.headDim, p.intermediateSize, p.vocabSize]
        guard dimensions.allSatisfy({ (1...1_000_000).contains($0) }),
              (1...256).contains(t.numLayers), (1...256).contains(p.numLayers),
              p.numCodeGroups == config.speechTokenizerDecoder.numQuantizers,
              [0, 4, 8].contains(t.bits), [0, 4, 8].contains(p.bits),
              [32, 64, 128].contains(t.groupSize), [32, 64, 128].contains(p.groupSize),
              t.numHeads % t.numKVHeads == 0, p.numHeads % p.numKVHeads == 0,
              t.headDim % 2 == 0, p.headDim % 2 == 0 else {
            throw invalidWeights("unsupported or unsafe model dimensions")
        }
    }

    private static func validateCodecConfiguration(at url: URL) throws {
        // Codec allocation follows the fixed architecture in this package. Some
        // released config fields are stale, so do not reinterpret those dimensions.
        // Still require an intact config for the correct codec family.
        do {
            let data = try Data(contentsOf: url)
            guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                  object["model_type"] as? String == "qwen3_tts_tokenizer_12hz",
                  object["decoder_config"] is [String: Any],
                  object["encoder_config"] is [String: Any] else {
                throw invalidWeights("invalid codec configuration")
            }
        } catch {
            throw invalidWeights("codec configuration cannot be resolved")
        }
    }

    static func invalidWeights(_ reason: String) -> TTSWeightValidationError {
        TTSWeightValidationError(reason: reason)
    }

    static func readWeightHeaders(from directory: URL) throws -> TTSWeightHeaders {
        let files = try FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: [.isRegularFileKey])
            .filter { $0.pathExtension == "safetensors" }.sorted { $0.lastPathComponent < $1.lastPathComponent }
        guard !files.isEmpty else { throw invalidWeights("no safetensors files") }
        var result: TTSWeightHeaders = [:]
        for file in files {
            guard try file.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile == true else {
                throw invalidWeights("weight shard is not a regular file")
            }
            let handle = try FileHandle(forReadingFrom: file)
            defer { try? handle.close() }
            let size = try handle.seekToEnd()
            try handle.seek(toOffset: 0)
            guard let prefix = try handle.read(upToCount: 8), prefix.count == 8 else {
                throw invalidWeights("truncated safetensors prefix")
            }
            let headerSize = prefix.enumerated().reduce(UInt64(0)) { $0 | UInt64($1.element) << ($1.offset * 8) }
            guard headerSize > 0, headerSize <= 64 * 1024 * 1024, size >= 8,
                  headerSize <= size - 8,
                  let data = try handle.read(upToCount: Int(headerSize)), data.count == Int(headerSize),
                  let entries = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                throw invalidWeights("invalid safetensors header")
            }
            let payloadSize = size - 8 - headerSize
            var extents: [(UInt64, UInt64)] = []
            for (name, raw) in entries where name != "__metadata__" {
                guard result[name] == nil else { throw invalidWeights("duplicate tensor \(name)") }
                guard let entry = raw as? [String: Any], let dtype = entry["dtype"] as? String,
                      let shape = integerList(entry["shape"]), let offsets = integerList(entry["data_offsets"]),
                      offsets.count == 2, offsets[0] >= 0, offsets[1] >= offsets[0],
                      shape.allSatisfy({ $0 > 0 }),
                      let width = ["F16": 2, "BF16": 2, "F32": 4, "F64": 8,
                                   "U32": 4, "I32": 4, "I64": 8, "U64": 8,
                                   "U8": 1, "I8": 1, "BOOL": 1][dtype] else {
                    throw invalidWeights("invalid tensor header \(name)")
                }
                var bytes = width
                for dimension in shape {
                    let product = bytes.multipliedReportingOverflow(by: dimension)
                    guard !product.overflow else { throw invalidWeights("tensor size overflow \(name)") }
                    bytes = product.partialValue
                }
                guard offsets[1] - offsets[0] == bytes, UInt64(offsets[1]) <= payloadSize else {
                    throw invalidWeights("invalid tensor extent \(name)")
                }
                extents.append((UInt64(offsets[0]), UInt64(offsets[1])))
                result[name] = TTSWeightTensor(shape: shape, dtype: dtype)
            }
            var end: UInt64 = 0
            for extent in extents.sorted(by: { $0.0 < $1.0 }) {
                guard extent.0 == end else { throw invalidWeights("overlapping or missing tensor payload") }
                end = extent.1
            }
            guard !extents.isEmpty, end == payloadSize else {
                throw invalidWeights("empty or trailing safetensors payload")
            }
        }
        return result
    }

    private static func integerList(_ raw: Any?) -> [Int]? {
        guard let values = raw as? [NSNumber] else { return nil }
        var result: [Int] = []
        for value in values {
            // Reject booleans, fractions and numbers outside Int's exact range.
            guard CFGetTypeID(value) != CFBooleanGetTypeID(),
                  let integer = Int(value.stringValue) else { return nil }
            result.append(integer)
        }
        return result
    }

    static func validateTalkerWeights(_ weights: TTSWeightHeaders, talker: TalkerConfig,
                                     predictor: CodePredictorConfig) throws {
        var schema = WeightSchema(weights)
        try schema.float("talker.model.codec_embedding.weight", [talker.codecVocabSize, talker.hiddenSize])
        let embedding = "talker.model.text_embedding"
        if let packed = weights[embedding + ".weight"], packed.dtype == "U32" {
            guard packed.shape.count == 2, packed.shape[1] <= talker.textHiddenSize,
                  let scales = weights[embedding + ".scales"], scales.shape.count == 2,
                  scales.shape[1] > 0, talker.textHiddenSize % scales.shape[1] == 0 else {
                throw invalidWeights("invalid quantized text embedding")
            }
            let bits = packed.shape[1] * 32 / talker.textHiddenSize
            try schema.linear(embedding, talker.textHiddenSize, talker.textVocabSize,
                              bits: bits, group: talker.textHiddenSize / scales.shape[1])
        } else {
            try schema.linear(embedding, talker.textHiddenSize, talker.textVocabSize, bits: 0, group: 1)
        }
        try schema.linear("talker.text_projection.linear_fc1", talker.textHiddenSize, talker.textHiddenSize,
                          bits: talker.bits, group: talker.groupSize, bias: true)
        try schema.linear("talker.text_projection.linear_fc2", talker.textHiddenSize, talker.hiddenSize,
                          bits: talker.bits, group: talker.groupSize, bias: true)
        try schema.linear("talker.codec_head", talker.hiddenSize, talker.codecVocabSize,
                          bits: talker.bits, group: talker.groupSize)
        try schema.float("talker.model.norm.weight", [talker.hiddenSize])
        try schema.transformer("talker.model", layers: talker.numLayers, hidden: talker.hiddenSize,
                               heads: talker.numHeads, kvHeads: talker.numKVHeads, headDim: talker.headDim,
                               intermediate: talker.intermediateSize, bits: talker.bits, group: talker.groupSize)
        let cp = "talker.code_predictor"
        for i in 0..<(predictor.numCodeGroups - 1) {
            try schema.float("\(cp).model.codec_embedding.\(i).weight", [predictor.vocabSize, predictor.embeddingDim])
            try schema.linear("\(cp).lm_head.\(i)", predictor.hiddenSize, predictor.vocabSize,
                              bits: predictor.bits, group: predictor.groupSize)
        }
        try schema.float("\(cp).model.norm.weight", [predictor.hiddenSize])
        try schema.transformer("\(cp).model", layers: predictor.numLayers, hidden: predictor.hiddenSize,
                               heads: predictor.numHeads, kvHeads: predictor.numKVHeads, headDim: predictor.headDim,
                               intermediate: predictor.intermediateSize, bits: predictor.bits, group: predictor.groupSize)
        if predictor.needsProjection {
            try schema.linear("\(cp).small_to_mtp_projection", predictor.embeddingDim, predictor.hiddenSize,
                              bits: predictor.bits, group: predictor.groupSize, bias: true)
        }
        try schema.uniformFloatType()
    }

    static func validateSpeakerWeights(_ weights: TTSWeightHeaders, embeddingDim: Int) throws {
        var schema = WeightSchema(weights)
        func conv(_ key: String, _ input: Int, _ output: Int, _ kernel: Int) throws {
            try schema.float("speaker_encoder.\(key).weight", [output, kernel, input],
                             alternative: [output, input, kernel])
            try schema.float("speaker_encoder.\(key).bias", [output])
        }
        try conv("blocks.0.conv", 128, 512, 5)
        for i in 1...3 {
            try conv("blocks.\(i).tdnn1.conv", 512, 512, 1)
            try conv("blocks.\(i).tdnn2.conv", 512, 512, 1)
            for j in 0..<7 { try conv("blocks.\(i).res2net_block.blocks.\(j).conv", 64, 64, 3) }
            try conv("blocks.\(i).se_block.conv1", 512, 128, 1)
            try conv("blocks.\(i).se_block.conv2", 128, 512, 1)
        }
        try conv("mfa.conv", 1536, 1536, 1)
        try conv("asp.tdnn.conv", 4608, 128, 1)
        try conv("asp.conv", 128, 1536, 1)
        try conv("fc", 3072, embeddingDim, 1)
        try schema.uniformFloatType()
    }

    /// Returns true only for a complete, shape-correct, uniformly F16 decoder.
    static func validateDecoderWeights(_ weights: TTSWeightHeaders,
                                      config c: SpeechTokenizerDecoderConfig) throws -> Bool {
        var s = WeightSchema(weights)
        for (name, count, size) in [("rvq_first", 1, c.semanticCodebookSize),
                                    ("rvq_rest", c.numQuantizers - 1, c.acousticCodebookSize)] {
            let base = "decoder.quantizer.\(name)"
            try s.conv(base + ".output_proj", c.codebookDim, c.hiddenSize, 1, bias: false)
            for i in 0..<count {
                let key = base + ".vq.layers.\(i)._codebook"
                if weights[key + ".embed"] != nil {
                    try s.float(key + ".embed", [size, c.codebookDim])
                } else {
                    try s.float(key + ".cluster_usage", [size])
                    try s.float(key + ".embedding_sum", [size, c.codebookDim])
                }
            }
        }
        try s.conv("decoder.pre_conv.conv", c.hiddenSize, c.latentDim, 3)
        let t = "decoder.pre_transformer"
        try s.linear(t + ".input_proj", c.latentDim, c.hiddenSize, bias: true)
        try s.linear(t + ".output_proj", c.hiddenSize, c.latentDim, bias: true)
        try s.float(t + ".norm.weight", [c.hiddenSize])
        for i in 0..<c.numLayers {
            let p = t + ".layers.\(i)"
            for proj in ["q_proj", "k_proj", "v_proj"] {
                try s.linear(p + ".self_attn." + proj, c.hiddenSize, c.numHeads * c.headDim)
            }
            try s.linear(p + ".self_attn.o_proj", c.numHeads * c.headDim, c.hiddenSize)
            for norm in ["input_layernorm", "post_attention_layernorm"] {
                try s.float(p + "." + norm + ".weight", [c.hiddenSize])
            }
            for proj in ["gate_proj", "up_proj"] { try s.linear(p + ".mlp." + proj, c.hiddenSize, c.hiddenSize * 2) }
            try s.linear(p + ".mlp.down_proj", c.hiddenSize * 2, c.hiddenSize)
            for scale in ["self_attn_layer_scale", "mlp_layer_scale"] {
                try s.float(p + "." + scale + ".scale", [c.hiddenSize])
            }
        }
        guard c.upsamplingRatios.count == 2 else { throw invalidWeights("decoder requires two pre-upsample stages") }
        for i in 0..<2 {
            let p = "decoder.upsample.\(i)"
            try s.conv(p + ".0.conv", c.latentDim, c.latentDim, c.upsamplingRatios[i], transposed: true)
            try s.conv(p + ".1.dwconv.conv", 1, c.latentDim, 7)
            try s.float(p + ".1.norm.weight", [c.latentDim])
            try s.float(p + ".1.norm.bias", [c.latentDim])
            try s.float(p + ".1.gamma", [c.latentDim])
            try s.linear(p + ".1.pwconv1", c.latentDim, c.latentDim * 4, bias: true)
            try s.linear(p + ".1.pwconv2", c.latentDim * 4, c.latentDim, bias: true)
        }
        try s.conv("decoder.decoder.0.conv", c.latentDim, c.decoderDim, 7)
        var dim = c.decoderDim
        for (i, rate) in c.upsampleRates.enumerated() {
            let p = "decoder.decoder.\(i + 1).block"
            try s.snake(p + ".0", dim)
            try s.conv(p + ".1.conv", dim, dim / 2, rate * 2, transposed: true)
            dim /= 2
            for j in 2...4 {
                try s.snake(p + ".\(j).act1", dim)
                try s.snake(p + ".\(j).act2", dim)
                try s.conv(p + ".\(j).conv1.conv", dim, dim, 7)
                try s.conv(p + ".\(j).conv2.conv", dim, dim, 1)
            }
        }
        try s.snake("decoder.decoder.5", dim)
        try s.conv("decoder.decoder.6.conv", dim, 1, 7)
        try s.uniformFloatType()
        // Every convolution was required and checked above. Never classify a
        // partially populated decoder by one (for example, the final) weight.
        return !s.convWeights.isEmpty && s.convWeights.allSatisfy { weights[$0]?.dtype == "F16" }
    }

    static func validateEncoderWeights(_ weights: TTSWeightHeaders) throws {
        var s = WeightSchema(weights)
        let c = MimiEncoderConfig()
        let base = "encoder.encoder.layers"
        try s.conv(base + ".0.conv", c.channels, c.nFilters, c.kSize)
        var dim = c.nFilters
        for (i, rate) in c.ratios.reversed().enumerated() {
            let p = base + ".\(1 + 3 * i).block"
            try s.conv(p + ".1.conv", dim, dim / 2, c.residualKSize)
            try s.conv(p + ".3.conv", dim / 2, dim, 1)
            try s.conv(base + ".\(3 + 3 * i).conv", dim, dim * 2, rate * 2)
            dim *= 2
        }
        try s.conv(base + ".14.conv", dim, c.dimension, c.lastKSize)
        for i in 0..<c.numLayers {
            let p = "encoder.encoder_transformer.layers.\(i)"
            for norm in ["input_layernorm", "post_attention_layernorm"] {
                try s.float(p + "." + norm + ".weight", [c.dimension])
                try s.float(p + "." + norm + ".bias", [c.dimension])
            }
            for proj in ["q_proj", "k_proj", "v_proj", "o_proj"] {
                try s.linear(p + ".self_attn." + proj, c.dimension, c.dimension)
            }
            try s.linear(p + ".mlp.fc1", c.dimension, c.intermediateSize)
            try s.linear(p + ".mlp.fc2", c.intermediateSize, c.dimension)
            for scale in ["self_attn_layer_scale", "mlp_layer_scale"] {
                try s.float(p + "." + scale + ".scale", [c.dimension])
            }
        }
        try s.conv("encoder.downsample.conv", c.dimension, c.dimension, c.compress * 2, bias: false)
        for (name, count) in [("semantic", 1), ("acoustic", c.validNumQuantizers - 1)] {
            let p = "encoder.quantizer.\(name)_residual_vector_quantizer"
            try s.conv(p + ".input_proj", c.dimension, c.codebookDim, 1, bias: false)
            for i in 0..<count {
                try s.float(p + ".layers.\(i).codebook.embed_sum", [c.codebookSize, c.codebookDim])
                try s.float(p + ".layers.\(i).codebook.cluster_usage", [c.codebookSize])
            }
        }
        try s.uniformFloatType()
    }
}

private struct WeightSchema {
    let weights: TTSWeightHeaders
    var floatTypes: Set<String> = []
    var convWeights: [String] = []
    init(_ weights: TTSWeightHeaders) { self.weights = weights }

    mutating func float(_ key: String, _ shape: [Int], alternative: [Int]? = nil) throws {
        guard let value = weights[key] else { throw TTSWeightLoader.invalidWeights("missing tensor \(key)") }
        guard value.shape == shape || value.shape == alternative,
              ["F16", "BF16", "F32"].contains(value.dtype) else {
            throw TTSWeightLoader.invalidWeights("shape or dtype mismatch \(key)")
        }
        floatTypes.insert(value.dtype)
    }

    func uniformFloatType() throws {
        guard floatTypes.count == 1 else { throw TTSWeightLoader.invalidWeights("mixed floating-point types in component") }
    }

    mutating func linear(_ prefix: String, _ input: Int, _ output: Int,
                         bits: Int = 0, group: Int = 1, bias: Bool = false) throws {
        if bits == 0 {
            guard weights[prefix + ".scales"] == nil, weights[prefix + ".biases"] == nil else {
                throw TTSWeightLoader.invalidWeights("unexpected quantization tensors \(prefix)")
            }
            try float(prefix + ".weight", [output, input])
        } else {
            guard [4, 8].contains(bits), group > 0, input % group == 0, input % (32 / bits) == 0,
                  let packed = weights[prefix + ".weight"], packed.dtype == "U32",
                  packed.shape == [output, input / (32 / bits)] else {
                throw TTSWeightLoader.invalidWeights("quantized weight mismatch \(prefix)")
            }
            try float(prefix + ".scales", [output, input / group])
            try float(prefix + ".biases", [output, input / group])
        }
        if bias { try float(prefix + ".bias", [output]) }
    }

    mutating func conv(_ prefix: String, _ input: Int, _ output: Int, _ kernel: Int,
                       bias: Bool = true, transposed: Bool = false) throws {
        try float(prefix + ".weight", transposed ? [input, output, kernel] : [output, input, kernel])
        convWeights.append(prefix + ".weight")
        if bias { try float(prefix + ".bias", [output]) }
    }

    mutating func snake(_ prefix: String, _ dim: Int) throws {
        try float(prefix + ".alpha", [dim])
        try float(prefix + ".beta", [dim])
    }

    mutating func transformer(_ prefix: String, layers: Int, hidden: Int, heads: Int,
                              kvHeads: Int, headDim: Int, intermediate: Int, bits: Int, group: Int) throws {
        for i in 0..<layers {
            let p = prefix + ".layers.\(i)"
            try linear(p + ".self_attn.q_proj", hidden, heads * headDim, bits: bits, group: group)
            try linear(p + ".self_attn.k_proj", hidden, kvHeads * headDim, bits: bits, group: group)
            try linear(p + ".self_attn.v_proj", hidden, kvHeads * headDim, bits: bits, group: group)
            try linear(p + ".self_attn.o_proj", heads * headDim, hidden, bits: bits, group: group)
            try float(p + ".self_attn.q_norm.weight", [headDim])
            try float(p + ".self_attn.k_norm.weight", [headDim])
            try float(p + ".input_layernorm.weight", [hidden])
            try float(p + ".post_attention_layernorm.weight", [hidden])
            try linear(p + ".mlp.gate_proj", hidden, intermediate, bits: bits, group: group)
            try linear(p + ".mlp.up_proj", hidden, intermediate, bits: bits, group: group)
            try linear(p + ".mlp.down_proj", intermediate, hidden, bits: bits, group: group)
        }
    }
}
