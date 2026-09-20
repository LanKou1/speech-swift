#if canImport(CoreML)
import CoreML
import Foundation

/// CoreML batch speech decoder: 16 codebooks → 24kHz audio.
/// Uses the bundle's fixed frame capacity. Pads shorter sequences and rejects overflow.
final class SpeechDecoderCoreML {
    private let model: MLModel
    private let batchFrames: Int
    private let samplesPerFrame = 1920

    init(model: MLModel, batchFrames: Int = 125) { self.model = model; self.batchFrames = batchFrames }

    /// Decode codebook indices to audio.
    /// - Parameter codes: [16][T] — 16 codebook indices for T frames
    /// - Returns: Audio samples at 24kHz, mono Float32
    func decode(codes: [[Int32]]) throws -> [Float] {
        let numFrames = try Self.validate(codes: codes, capacity: batchFrames)

        // Build the fixed-size [1, 16, batchFrames] input, zero-padded
        let input = try MLMultiArray(shape: [1, 16, NSNumber(value: batchFrames)], dataType: .int32)
        let ptr = input.dataPointer.assumingMemoryBound(to: Int32.self)
        memset(ptr, 0, 16 * batchFrames * 4)
        for cb in 0..<16 {
            for t in 0..<numFrames {
                ptr[cb * batchFrames + t] = codes[cb][t]
            }
        }

        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "audio_codes": MLFeatureValue(multiArray: input)])
        let result = try model.prediction(from: provider)
        let audioArray = result.featureValue(for: "audio")!.multiArrayValue!

        // Extract and trim to actual frame count
        let totalSamples = min(numFrames * samplesPerFrame, audioArray.count)
        var audio = [Float](repeating: 0, count: totalSamples)
        if audioArray.dataType == .float16 {
            let src = audioArray.dataPointer.assumingMemoryBound(to: Float16.self)
            for i in 0..<totalSamples { audio[i] = Float(src[i]) }
        } else {
            let src = audioArray.dataPointer.assumingMemoryBound(to: Float.self)
            for i in 0..<totalSamples { audio[i] = src[i] }
        }
        return audio
    }

    static func validate(codes: [[Int32]], capacity: Int) throws -> Int {
        let numCodebooks = codes.count
        let numFrames = codes.first?.count ?? 0
        guard numCodebooks == 16, numFrames > 0, numFrames <= capacity,
              codes.allSatisfy({ $0.count == numFrames && $0.allSatisfy { (0..<2048).contains($0) } }) else {
            throw SpeechDecoderError.invalidInput("Expected 16 equally sized codebooks with 1...\(capacity) frames and tokens in 0..<2048")
        }

        return numFrames
    }

    enum SpeechDecoderError: Error {
        case invalidInput(String)
    }
}
#endif
