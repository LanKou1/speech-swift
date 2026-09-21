// Standalone checkpoint-backed regression candidate. No private voice assets.
// Usage: decoder-length-probe /path/to/speech_tokenizer
import Foundation
import MLX
import Qwen3TTS

@main struct DecoderLengthProbe {
    static func main() throws {
        guard CommandLine.arguments.count == 2 else {
            throw NSError(domain: "DecoderLengthProbe", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "Pass a local speech tokenizer directory"])
        }
        Memory.cacheLimit = 256 * 1024 * 1024
        let decoder = SpeechTokenizerDecoder(config: SpeechTokenizerDecoderConfig())
        try TTSWeightLoader.loadSpeechTokenizerDecoderWeights(
            into: decoder,
            from: URL(fileURLWithPath: CommandLine.arguments[1]))
        decoder.setupCompilation()
        var failures = 0
        // Covers short streaming-sized inputs, the direct/chunked boundary,
        // multiple chunks, and a tiny final chunk. Codes are synthetic, not speech.
        for frames in [4, 12, 25, 35, 36, 50, 51, 76, 384] {
            let codes = MLXArray.ones([1, 16, frames], dtype: .int32)
            let waveform = decoder.chunkedDecode(codes: codes)
            eval(waveform)
            let expected = frames * 1920
            let actual = waveform.dim(1)
            if actual != expected { failures += 1 }
            print("frames=\(frames) expected=\(expected) actual=\(actual) pass=\(actual == expected)")
        }
        if failures > 0 {
            throw NSError(domain: "DecoderLengthProbe", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "\(failures) decoder length assertions failed"])
        }
    }
}
