// Compile against Qwen3TTS and MLX. No personal assets are bundled.
// Usage: probe /absolute/path/profile.json /absolute/path/output-directory
// The output contains synthetic target speech in the supplied reference voice.
import Foundation
import MLX
import AudioCommon
import Qwen3TTS

private struct Profile: Decodable {
    let modelId: String, modelDirectory: String, codecDirectory: String
    let referenceFilename: String, referenceText: String
}

@main struct ProgressiveCloneProbe {
    static func main() async throws {
        guard CommandLine.arguments.count == 3 else {
            throw NSError(domain: "Usage: probe profile.json output-directory", code: 2)
        }
        let profileURL = URL(fileURLWithPath: CommandLine.arguments[1])
        let output = URL(fileURLWithPath: CommandLine.arguments[2], isDirectory: true)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        let profile = try JSONDecoder().decode(Profile.self, from: Data(contentsOf: profileURL))
        let reference = try AudioFileLoader.load(url: profileURL.deletingLastPathComponent().appendingPathComponent(profile.referenceFilename), targetSampleRate: 24000)
        Memory.cacheLimit = 256 * 1024 * 1024
        let (model, encoder) = try await Qwen3TTSModel.fromPretrainedWithEncoder(
            modelId: profile.modelId, cacheDir: URL(fileURLWithPath: profile.modelDirectory),
            tokenizerCacheDir: URL(fileURLWithPath: profile.codecDirectory), offlineMode: true)
        let sampling = SamplingConfig(temperature: 0.7, topK: 50, topP: 1, repetitionPenalty: 1.5, maxTokens: 256)
        let fixtures = ["今天有点凉，出门记得带件外套。", "没关系，我们慢慢来，一步一步就好。", "缓存就是先把用过的东西存起来，下次就能更快打开。", "你好。", "好，那就改成明天下午三点半，还是和 Alex 在 Zoom 开会。"]
        var results: [[String: Any]] = []
        for (index, text) in fixtures.enumerated() {
            var baseline: [Float] = [], final: [Float] = [], emitted: [Float] = []
            var baselineSeconds = 0.0, progressiveSeconds = 0.0
            var packets: [(Double, Int)] = []
            for progressive in index % 2 == 0 ? [false, true] : [true, false] {
                model.clearReferenceAudioCache()
                MLXRandom.seed(42)
                let start = Date()
                if progressive {
                    final = try model.synthesizeWithVoiceCloneICLProgressively(
                        text: text, referenceAudio: reference, referenceText: profile.referenceText,
                        language: "chinese", sampling: sampling, codecEncoder: encoder, onAudio: { chunk in
                            packets.append((Date().timeIntervalSince(start), chunk.count))
                            emitted.append(contentsOf: chunk)
                        })
                    progressiveSeconds = Date().timeIntervalSince(start)
                } else {
                    baseline = try model.synthesizeWithVoiceCloneICLCancellable(
                        text: text, referenceAudio: reference, referenceText: profile.referenceText,
                        language: "chinese", sampling: sampling, codecEncoder: encoder)
                    baselineSeconds = Date().timeIntervalSince(start)
                }
            }
            guard !baseline.isEmpty, baseline == final, emitted.count == baseline.count,
                  emitted.allSatisfy(\.isFinite), let first = packets.first else {
                throw NSError(domain: "Generation parity or missing PCM fixture \(index)", code: 1)
            }
            let differences = zip(baseline, emitted).map { Double($0 - $1) }
            let rms = sqrt(differences.reduce(0) { $0 + $1 * $1 } / Double(differences.count))
            let peak = differences.map(abs).max() ?? 0
            guard rms <= 0.0001, peak <= 0.01 else {
                throw NSError(domain: "Prefix waveform parity fixture \(index)", code: 1)
            }
            var queued = 0, underrun = 0.0
            for (time, samples) in packets {
                if queued > 0 { underrun = max(underrun, time - first.0 - Double(queued) / 24000) }
                queued += samples
            }
            for (name, samples) in [("baseline",baseline),("emitted",emitted)] {
                try samples.withUnsafeBytes { try Data($0).write(to: output.appendingPathComponent("\(index)-\(name).f32")) }
            }
            results.append(["fixture": index, "baselineSeconds": baselineSeconds,
                            "firstPacketSeconds": first.0, "progressiveSeconds": progressiveSeconds,
                            "rms": rms, "peak": peak, "predictedUnderrunSeconds": underrun,
                            "samples": emitted.count,
                            "packets": packets.map { ["seconds": $0.0, "samples": Double($0.1)] }])
        }
        let cancelled = Task { () -> Bool in
            var packets = 0
            do {
                _ = try model.synthesizeWithVoiceCloneICLProgressively(
                    text: fixtures[2], referenceAudio: reference, referenceText: profile.referenceText,
                    language: "chinese", sampling: sampling, codecEncoder: encoder, onAudio: { _ in
                        packets += 1
                        withUnsafeCurrentTask { $0?.cancel() }
                    })
                return false
            } catch is CancellationError { return packets == 1 }
            catch { return false }
        }
        let cancellationPassed = await cancelled.value
        let result: [String: Any] = ["fixtures": results, "cancellationAfterFirstPacket": cancellationPassed,
                                    "mlxPeakBytes": Memory.peakMemory,
                                    "limits": "Standalone native packets, not playback or simultaneous ASR/SLM residency. Numerical parity does not replace listening. Timing and inventory are measurements, not portable speed assertions."]
        try JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted,.sortedKeys]).write(to: output.appendingPathComponent("result.json"))
        guard cancellationPassed else { throw NSError(domain: "Cancellation failed", code: 1) }
        print("PROGRESSIVE_CLONE_PROBE: 5 waveform pairs and cancellation passed; inspect timing and inventory in result.json")
    }
}
