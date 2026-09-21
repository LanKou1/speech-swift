// Usage: prefix-probe /path/to/speech_tokenizer /path/to/codes.npy
// Codes file is [1,time,16]. Neither checkpoint nor audio assets are bundled.
import Foundation
import MLX
import Qwen3TTS

private enum ProbeFailure: Error { case mismatch, cancelled }

@main struct PrefixDecodeProbe {
    static func main() throws {
        guard CommandLine.arguments.count == 3 else {
            throw NSError(domain: "PrefixDecodeProbe", code: 2,
                userInfo: [NSLocalizedDescriptionKey: "Pass local codec directory and codes.npy"])
        }
        Memory.cacheLimit = 256 * 1024 * 1024
        let decoder = SpeechTokenizerDecoder(config: SpeechTokenizerDecoderConfig())
        try TTSWeightLoader.loadSpeechTokenizerDecoderWeights(
            into: decoder, from: URL(fileURLWithPath: CommandLine.arguments[1]))
        decoder.setupCompilation()
        let codes = try loadArray(url: URL(fileURLWithPath: CommandLine.arguments[2]))
            .transposed(0, 2, 1).asType(.int32)
        precondition(codes.dim(2) >= 383)
        var checks = 0
        for frames in [35, 301, 325, 326, 383] {
            let input = codes[0..., 0..., ..<frames]
            let full = decoder.chunkedDecode(codes: input, chunkSize: 300, leftContext: 25)
            eval(full)
            precondition(full.dim(1) == frames * 1920)
            let starts = Array(Set([0, 24, 25, 299, 300, 301, 349, frames]
                .filter { $0 <= frames })).sorted()
            for start in starts {
                let trimmed = decoder.chunkedDecode(
                    codes: input, chunkSize: 300, leftContext: 25, startFrame: start)
                eval(trimmed)
                let expected = full[0..., (start * 1920)..., 0...]
                let same = trimmed.shape == expected.shape
                    && arrayEqual(trimmed, expected).item(Bool.self)
                print("frames=\(frames) start=\(start) samples=\(trimmed.dim(1)) exact=\(same)")
                guard same else { throw ProbeFailure.mismatch }
                checks += 1
            }
        }
        // Cancellation must be observed before the first skipped chunk and
        // at the next iteration before a real decode. It also applies to empty output.
        let longInput = codes[0..., 0..., ..<383]
        for (start, throwAt) in [(349, 1), (349, 2), (383, 1)] {
            var calls = 0
            do {
                _ = try decoder.chunkedDecode(
                    codes: longInput, chunkSize: 300, leftContext: 25,
                    startFrame: start, checkCancellation: {
                        calls += 1
                        if calls == throwAt { throw ProbeFailure.cancelled }
                    })
                throw ProbeFailure.mismatch
            } catch ProbeFailure.cancelled {
                guard calls == throwAt else { throw ProbeFailure.mismatch }
                print("cancel start=\(start) checkpoint=\(throwAt) pass=true")
                checks += 1
            }
        }
        for count in [0, 1, 255, 256, 257] {
            let mel = SpeakerMel.compute(audio: [Float](repeating: 0.1, count: count))
            eval(mel)
            guard mel.shape == [1, 1, 128], mel.asArray(Float.self).allSatisfy({$0.isFinite}) else { throw ProbeFailure.mismatch }
            print("short mel samples=\(count) finite=true frames=1")
            checks += 1
        }
        print("PASS \(checks) checks")
    }
}
