import Foundation

/// Dimensions of a six-component Qwen3-TTS CoreML bundle.
struct BundleConfiguration: Decodable {
    let hiddenSize: Int
    let maxSequenceLength: Int
    let speechDecoderFrames: Int
    let predictorKVDimension: Int
    let requiresSpeakerEmbedding: Bool

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size", maxSequenceLength = "max_seq_len"
        case speechDecoderFrames = "speech_decoder_frames", legacyFrames = "max_codec_tokens"
        case predictorKVDimension = "predictor_kv_dim"
        case requiresSpeakerEmbedding = "requires_speaker_embedding"
        case sampleRate = "sample_rate", samplesPerFrame = "samples_per_frame"
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1024
        maxSequenceLength = try c.decodeIfPresent(Int.self, forKey: .maxSequenceLength) ?? 256
        speechDecoderFrames = try c.decodeIfPresent(Int.self, forKey: .speechDecoderFrames)
            ?? c.decodeIfPresent(Int.self, forKey: .legacyFrames) ?? 125
        predictorKVDimension = try c.decodeIfPresent(Int.self, forKey: .predictorKVDimension) ?? 5120
        requiresSpeakerEmbedding = try c.decodeIfPresent(Bool.self, forKey: .requiresSpeakerEmbedding) ?? false
        let rate = try c.decodeIfPresent(Int.self, forKey: .sampleRate) ?? 24000
        let samples = try c.decodeIfPresent(Int.self, forKey: .samplesPerFrame) ?? 1920
        guard [1024, 2048].contains(hiddenSize), (1...32768).contains(maxSequenceLength),
              (1...32768).contains(speechDecoderFrames), (1...131072).contains(predictorKVDimension),
              rate == 24000, samples == 1920 else {
            throw BundleError.invalidConfiguration("Unsupported dimensions or audio rate")
        }
    }

    static func load(from directory: URL) throws -> Self {
        let url = directory.appendingPathComponent("config.json")
        let data = FileManager.default.fileExists(atPath: url.path) ? try Data(contentsOf: url) : Data("{}".utf8)
        return try JSONDecoder().decode(Self.self, from: data)
    }

    func generationLimit(requested: Int, promptCount: Int) throws -> Int {
        guard requested > 0, requested <= speechDecoderFrames else {
            throw BundleError.invalidInput("maxTokens must be in 1...\(speechDecoderFrames), the exported SpeechDecoder capacity")
        }
        guard promptCount > 0, promptCount < maxSequenceLength else {
            throw BundleError.invalidInput("Prompt must leave room in the \(maxSequenceLength)-position cache")
        }
        // Preserve the original 0.6B generation budget while sizing it from the bundle.
        return min(requested, min(8 * promptCount, maxSequenceLength - promptCount))
    }
}

enum BundleError: LocalizedError {
    case invalidConfiguration(String)
    case invalidInput(String)
    var errorDescription: String? {
        switch self {
        case .invalidConfiguration(let reason): return "Invalid Qwen3-TTS CoreML bundle: \(reason)"
        case .invalidInput(let reason): return "Invalid Qwen3-TTS CoreML input: \(reason)"
        }
    }
}

/// Read the little-endian Float32 vectors emitted by the public exporter.
/// Validate the header and payload before exposing embeddings to CoreML.
struct EmbeddingFile {
    static func read(_ data: Data, channels: Int) throws -> [Float] {
        func invalid() -> BundleError { .invalidInput("Expected a finite \(channels)-channel Float32 NPY embedding") }
        guard data.count >= 10, Array(data.prefix(6)) == [0x93, 78, 85, 77, 80, 89] else { throw invalid() }
        let major = data[6]
        let start: Int, length: Int
        switch major {
        case 1: start = 10; length = Int(data[8]) | Int(data[9]) << 8
        case 2, 3:
            guard data.count >= 12 else { throw invalid() }
            start = 12
            length = Int(data[8]) | Int(data[9]) << 8 | Int(data[10]) << 16 | Int(data[11]) << 24
        default: throw invalid()
        }
        guard length <= data.count - start,
              let header = String(data: data.subdata(in: start..<(start + length)), encoding: .utf8) else { throw invalid() }
        func capture(_ pattern: String) -> String? {
            guard let r = try? NSRegularExpression(pattern: pattern),
                  let m = r.firstMatch(in: header, range: NSRange(header.startIndex..., in: header)),
                  let range = Range(m.range(at: 1), in: header) else { return nil }
            return String(header[range])
        }
        guard capture("['\"]descr['\"]\\s*:\\s*['\"]([^'\"]+)['\"]") == "<f4",
              capture("['\"]fortran_order['\"]\\s*:\\s*(True|False)") == "False",
              let shape = capture("['\"]shape['\"]\\s*:\\s*\\(([^)]*)\\)") else { throw invalid() }
        let dimensions = shape.split(separator: ",").map { Int($0.trimmingCharacters(in: .whitespaces)) }
        guard dimensions == [channels] || dimensions == [1, channels, 1, 1],
              data.count - start - length == channels * 4 else { throw invalid() }
        let values: [Float] = data.withUnsafeBytes { raw in
            (0..<channels).map { i in
                Float(bitPattern: UInt32(littleEndian: raw.loadUnaligned(fromByteOffset: start + length + i * 4, as: UInt32.self)))
            }
        }
        guard values.allSatisfy(\.isFinite) else { throw invalid() }
        return values
    }
}
