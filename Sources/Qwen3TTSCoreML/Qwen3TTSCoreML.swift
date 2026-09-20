#if canImport(CoreML)
import CoreML
import Foundation
import AudioCommon

/// Qwen3-TTS CoreML inference with six compiled model components.
///
/// Models: TextProjector, CodeEmbedder, MultiCodeEmbedder, CodeDecoder,
///         MultiCodeDecoder, SpeechDecoder
public final class Qwen3TTSCoreMLModel {
    public static let defaultModelId = "aufklarer/Qwen3-TTS-CoreML"
    public static let largeModelId = "aufklarer/Qwen3-TTS-1.7B-CoreML"

    private var codeDecoder: CodeDecoderInterface?
    private var multiCodeDecoder: MultiCodeDecoderInterface?
    private var speechDecoder: SpeechDecoderCoreML?
    private var textProjector: TextProjectorModel?
    private var codeEmbedder: CodeEmbedderModel?
    private var multiCodeEmbedder: MultiCodeEmbedderModel?
    private var tokenizer: Qwen3Tokenizer?

    // Pre-computed special embeddings [1, hiddenSize, 1, 1]
    private var ttsPadEmbed: MLMultiArray?
    private var ttsBosEmbed: MLMultiArray?
    private var ttsEosEmbed: MLMultiArray?
    public var speakerEmbedding: MLMultiArray?

    public private(set) var hiddenSize = 1024
    public private(set) var maxSequenceLength = 256
    public private(set) var maximumAudioFrames = 125
    private var configuration: BundleConfiguration?
    private let codecVocabSize = 3072
    private let codecEos = 2150

    /// Load a compiled bundle. The default model remains 0.6B.
    /// - Parameters:
    ///   - computeUnits: Decoder route; nil uses the bundle default (0.6B: ANE, 1.7B: CPU).
    ///   - speakerEmbeddingURL: Prepared Float32 NPY vector matching the talker width.
    ///     Required at synthesis time when the bundle has no default speaker.
    public static func fromPretrained(
        modelId: String = defaultModelId,
        localPath: String? = nil,
        cacheDir: URL? = nil,
        offlineMode: Bool = false,
        computeUnits: MLComputeUnits? = nil,
        speakerEmbeddingURL: URL? = nil,
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> Qwen3TTSCoreMLModel {
        let resolvedCacheDir: URL
        if let localPath {
            resolvedCacheDir = URL(fileURLWithPath: localPath, isDirectory: true)
        } else {
            resolvedCacheDir = try cacheDir ?? HuggingFaceDownloader.getCacheDirectory(for: modelId)
            progressHandler?(0.0, "Downloading model...")
            try await HuggingFaceDownloader.downloadWeights(
                modelId: modelId, to: resolvedCacheDir,
                additionalFiles: [
                    "TextProjector.mlmodelc/**", "CodeEmbedder.mlmodelc/**",
                    "MultiCodeEmbedder.mlmodelc/**", "CodeDecoder.mlmodelc/**",
                    "MultiCodeDecoder.mlmodelc/**", "SpeechDecoder.mlmodelc/**",
                    "speaker_embedding.npy", "tts_pad_embed.npy",
                    "tts_bos_embed.npy", "tts_eos_embed.npy",
                    "config.json", "vocab.json", "merges.txt",
                ],
                offlineMode: offlineMode
            ) { progress in progressHandler?(progress * 0.7, "Downloading model...") }
        }

        let configuration = try BundleConfiguration.load(from: resolvedCacheDir)

        // Embedders on CPU (FP32 precision for accumulation, matching TTSKit).
        let cpuConfig = MLModelConfiguration()
        cpuConfig.computeUnits = .cpuOnly

        // Preserve the legacy ANE default; the 1.7B FP32 bundle defaults to its
        // validated CPU route. Explicit computeUnits and environment overrides
        // allow other routes to be evaluated on the target device.
        let decoderRoute = computeUnits ?? (configuration.hiddenSize == 2048 ? .cpuOnly : .cpuAndNeuralEngine)
        func route(_ key: String, _ fallback: MLComputeUnits) -> MLModelConfiguration {
            let cfg = MLModelConfiguration()
            switch ProcessInfo.processInfo.environment[key] {
            case "ane": cfg.computeUnits = .cpuAndNeuralEngine
            case "gpu": cfg.computeUnits = .cpuAndGPU
            case "cpu": cfg.computeUnits = .cpuOnly
            case "all": cfg.computeUnits = .all
            default:    cfg.computeUnits = CoreMLComputeUnitsResolver.resolved(default: fallback)
            }
            return cfg
        }
        let cdConfig = route("QWEN3TTS_ROUTE_CD", decoderRoute)
        let mcdConfig = route("QWEN3TTS_ROUTE_MCD", decoderRoute)
        let sdConfig = route("QWEN3TTS_ROUTE_SD", decoderRoute)

        let defaultConfig = cdConfig

        let model = Qwen3TTSCoreMLModel()
        model.configuration = configuration
        model.hiddenSize = configuration.hiddenSize
        model.maxSequenceLength = configuration.maxSequenceLength
        model.maximumAudioFrames = configuration.speechDecoderFrames

        progressHandler?(0.7, "Loading models...")

        // Load 6 models. The HuggingFace repo ships only ``.mlmodelc`` — on-device
        // ``MLModel.compileModel`` is known to drift per runtime (Mac vs
        // simulator vs iPhone) so we never run it here.
        func loadML(_ name: String, _ cfg: MLModelConfiguration = defaultConfig) throws -> MLModel {
            let compiledURL = resolvedCacheDir.appendingPathComponent("\(name).mlmodelc", isDirectory: true)
            let loaded = try MLModel(contentsOf: compiledURL, configuration: cfg)
            func require(_ feature: String, _ shape: [Int]) throws {
                let actual = loaded.modelDescription.inputDescriptionsByName[feature]?.multiArrayConstraint?.shape.map(\.intValue)
                guard actual == shape else {
                    throw BundleError.invalidConfiguration("\(name).\(feature): expected \(shape), got \(actual ?? [])")
                }
            }
            if name == "CodeDecoder" || name == "MultiCodeDecoder" {
                try require("input_embeds", [1, configuration.hiddenSize, 1, 1])
                try require("key_padding_mask", [1, name == "CodeDecoder" ? configuration.maxSequenceLength : 16])
            }
            if name == "MultiCodeDecoder", loaded.modelDescription.stateDescriptionsByName.isEmpty {
                try require("key_cache", [1, configuration.predictorKVDimension, 1, 16])
                try require("value_cache", [1, configuration.predictorKVDimension, 1, 16])
            }
            if name == "SpeechDecoder" { try require("audio_codes", [1, 16, configuration.speechDecoderFrames]) }
            if ["TextProjector", "CodeEmbedder", "MultiCodeEmbedder"].contains(name) {
                let shape = loaded.modelDescription.outputDescriptionsByName["input_embeds"]?.multiArrayConstraint?.shape.map(\.intValue)
                guard shape == [configuration.hiddenSize, 1, 1] || shape == [1, configuration.hiddenSize, 1, 1] else {
                    throw BundleError.invalidConfiguration("\(name) embedding width disagrees with config.json")
                }
            }
            return loaded
        }

        model.textProjector = TextProjectorModel(model: try loadML("TextProjector", cpuConfig))
        model.codeEmbedder = CodeEmbedderModel(model: try loadML("CodeEmbedder", cpuConfig))
        model.multiCodeEmbedder = MultiCodeEmbedderModel(model: try loadML("MultiCodeEmbedder", cpuConfig))

        // Detect chunked decoder layouts emitted by `convert_coreml.py
        // --ane-recipe`. Each decoder ships as N stateless ≤4-layer chunks
        // plus a 1-of-a-kind head model. If the chunk artifacts are absent
        // we fall back to the legacy monolithic single-model export.
        let fm = FileManager.default
        let bundleContents = (try? fm.contentsOfDirectory(atPath: resolvedCacheDir.path)) ?? []

        // CD: prefer chunked layout when present.
        let cdChunkFiles = bundleContents
            .filter { $0.hasPrefix("CodeDecoder_chunk") && $0.hasSuffix(".mlmodelc") }
            .sorted()
        if !cdChunkFiles.isEmpty {
            guard configuration.hiddenSize == 1024, configuration.maxSequenceLength == 256 else {
                throw BundleError.invalidConfiguration("Chunked CodeDecoder requires the legacy 0.6B/256 layout")
            }
            let cdChunks: [MLModel] = try cdChunkFiles.map { name in
                let url = resolvedCacheDir.appendingPathComponent(name, isDirectory: true)
                return try MLModel(contentsOf: url, configuration: cdConfig)
            }
            let cdHeadURL = resolvedCacheDir.appendingPathComponent(
                "CodeDecoder_head.mlmodelc", isDirectory: true)
            let cdHead = try MLModel(contentsOf: cdHeadURL, configuration: cdConfig)
            model.codeDecoder = TalkerGeneratorChunked(chunks: cdChunks, head: cdHead)
        } else {
            model.codeDecoder = TalkerGenerator(model: try loadML("CodeDecoder", cdConfig), maxSeqLen: model.maxSequenceLength, hiddenSize: model.hiddenSize)
        }

        // MCD: same detection pattern.
        let chunkFiles = bundleContents
            .filter { $0.hasPrefix("MultiCodeDecoder_chunk") && $0.hasSuffix(".mlmodelc") }
            .sorted()
        if !chunkFiles.isEmpty {
            guard configuration.hiddenSize == 1024 else {
                throw BundleError.invalidConfiguration("1.7B requires the monolithic predictor with its input projection")
            }
            let chunks: [MLModel] = try chunkFiles.map { name in
                let url = resolvedCacheDir.appendingPathComponent(name, isDirectory: true)
                return try MLModel(contentsOf: url, configuration: mcdConfig)
            }
            let headURL = resolvedCacheDir.appendingPathComponent(
                "MultiCodeDecoder_head.mlmodelc", isDirectory: true)
            let headModel = try MLModel(contentsOf: headURL, configuration: mcdConfig)
            model.multiCodeDecoder = MultiCodeDecoderChunked(chunks: chunks, head: headModel)
        } else {
            model.multiCodeDecoder = MultiCodeDecoderCoreML(model: try loadML("MultiCodeDecoder", mcdConfig), inputWidth: model.hiddenSize, totalKVDim: configuration.predictorKVDimension)
        }

        model.speechDecoder = SpeechDecoderCoreML(model: try loadML("SpeechDecoder", sdConfig), batchFrames: model.maximumAudioFrames)

        progressHandler?(0.9, "Loading embeddings...")

        func loadNpy(_ url: URL) throws -> MLMultiArray {
            let values = try EmbeddingFile.read(Data(contentsOf: url), channels: model.hiddenSize)
            let result = try MLMultiArray(shape: [1, NSNumber(value: model.hiddenSize), 1, 1], dataType: .float32)
            let dst = result.dataPointer.assumingMemoryBound(to: Float.self)
            for (i, value) in values.enumerated() { dst[i] = value }
            return result
        }
        model.ttsPadEmbed = try loadNpy(resolvedCacheDir.appendingPathComponent("tts_pad_embed.npy"))
        model.ttsBosEmbed = try loadNpy(resolvedCacheDir.appendingPathComponent("tts_bos_embed.npy"))
        model.ttsEosEmbed = try loadNpy(resolvedCacheDir.appendingPathComponent("tts_eos_embed.npy"))
        let speakerURL = speakerEmbeddingURL ?? resolvedCacheDir.appendingPathComponent("speaker_embedding.npy")
        if speakerEmbeddingURL != nil || fm.fileExists(atPath: speakerURL.path) {
            model.speakerEmbedding = try loadNpy(speakerURL)
        }

        // Load tokenizer
        let tokenizer = Qwen3Tokenizer()
        let vocabURL = resolvedCacheDir.appendingPathComponent("vocab.json")
        if FileManager.default.fileExists(atPath: vocabURL.path) {
            try tokenizer.load(from: vocabURL)
        }
        model.tokenizer = tokenizer

        progressHandler?(1.0, "Ready")
        return model
    }

    // MARK: - Synthesis

    public func synthesize(
        text: String,
        language: String = "english",
        temperature: Float = 0.8,
        topK: Int = 50,
        maxTokens: Int = 125,
        repetitionPenalty: Float = 1.05
    ) throws -> [Float] {
        guard let codeDecoder, let multiCodeDecoder, let speechDecoder,
              let textProjector, let codeEmbedder, let multiCodeEmbedder,
              let tokenizer, let ttsPadEmbed, let ttsBosEmbed, let ttsEosEmbed, let configuration else {
            throw TTSCoreMLError.modelNotLoaded
        }

        guard temperature.isFinite, temperature >= 0, topK >= 0,
              repetitionPenalty.isFinite, repetitionPenalty > 0 else {
            throw BundleError.invalidInput("Sampling parameters must be finite and nonnegative; repetitionPenalty must be positive")
        }
        if configuration.requiresSpeakerEmbedding && speakerEmbedding == nil {
            throw BundleError.invalidInput("Supply a \(hiddenSize)-channel speaker embedding using speakerEmbeddingURL or speakerEmbedding")
        }
        if let speakerEmbedding {
            guard speakerEmbedding.shape.map(\.intValue) == [1, hiddenSize, 1, 1],
                  (0..<hiddenSize).allSatisfy({ speakerEmbedding[[0, NSNumber(value: $0), 0, 0]].floatValue.isFinite }) else {
                throw BundleError.invalidInput("Expected a finite [1, \(hiddenSize), 1, 1] speaker embedding")
            }
        }
        // Check the independent vocoder limit before any model prediction.
        _ = try configuration.generationLimit(requested: maxTokens, promptCount: 1)

        // Per-stage timing — printed at the end when QWEN3TTS_BENCH=1
        let benchOn = ProcessInfo.processInfo.environment["QWEN3TTS_BENCH"] == "1"
        let tStart = CFAbsoluteTimeGetCurrent()
        var tPromptEnd = tStart, tDecodeEnd = tStart, tVocoderEnd = tStart
        var cdPrefillCalls = 0, cdDecodeCalls = 0, mcdPredictCalls = 0
        var cdPrefillSec = 0.0, cdDecodeSec = 0.0, mcdSec = 0.0, embedSec = 0.0

        // Build non-streaming prefill (all text in prefill)
        let prefillEmbeds = try PromptBuilder.build(
            text: text, language: language, tokenizer: tokenizer,
            textProjector: textProjector, codeEmbedder: codeEmbedder,
            ttsPadEmbed: ttsPadEmbed, ttsBosEmbed: ttsBosEmbed,
            ttsEosEmbed: ttsEosEmbed, speakerEmbedding: speakerEmbedding, hiddenSize: hiddenSize)
        tPromptEnd = CFAbsoluteTimeGetCurrent()

        let effectiveMaxTokens = try configuration.generationLimit(requested: maxTokens, promptCount: prefillEmbeds.count)

        // Reset CodeDecoder KV cache
        codeDecoder.resetCache()

        // Prefill: run all positions through CodeDecoder
        var lastLogits = [Float]()
        var lastHidden = try MLMultiArray(shape: [1, NSNumber(value: hiddenSize), 1, 1], dataType: .float16)
        for embed in prefillEmbeds {
            let t = CFAbsoluteTimeGetCurrent()
            (lastLogits, _) = try codeDecoder.forward(embedArray: embed)
            cdPrefillSec += CFAbsoluteTimeGetCurrent() - t
            cdPrefillCalls += 1
        }
        lastHidden = codeDecoder.lastHiddenState!

        // Build suppress token set: [2048, 3072) except EOS (matching TTSKit)
        // Suppress EOS for first token (min_new_tokens=2)
        lastLogits[codecEos] = -1e9
        var nextToken = TTSSampler.sample(
            logits: lastLogits, temperature: temperature, topK: topK,
            suppressRange: (2048, 3072), eosTokenId: codecEos)

        var allCodebooks = (0..<16).map { _ in [Int32]() }
        allCodebooks[0].append(nextToken)
        var generatedCB0 = [nextToken]

        // MultiCodeDecoder: predict CB1-15 for first frame
        let tMcd0 = CFAbsoluteTimeGetCurrent()
        var cpTokens = try multiCodeDecoder.predict(
            hiddenState: lastHidden, cb0Token: nextToken,
            codeEmbedder: codeEmbedder, multiCodeEmbedder: multiCodeEmbedder,
            temperature: temperature, topK: topK)
        mcdSec += CFAbsoluteTimeGetCurrent() - tMcd0
        mcdPredictCalls += 1
        for (i, token) in cpTokens.enumerated() {
            allCodebooks[i + 1].append(token)
        }

        // Autoregressive decode loop (capped at effectiveMaxTokens)
        for step in 1..<effectiveMaxTokens {
            // Step input = sum(all 16 codec embeddings) + tts_pad
            // Accumulate in FP32 to match Python's numpy precision, cast to FP16 at the end
            var sum32 = [Float](repeating: 0, count: hiddenSize)

            // Add CB0 embedding
            let tEmb0 = CFAbsoluteTimeGetCurrent()
            let cb0Emb = ensureNCHW(try codeEmbedder.embed(Int(nextToken)), channels: hiddenSize)
            let cb0Ptr = cb0Emb.dataPointer.assumingMemoryBound(to: Float16.self)
            for i in 0..<hiddenSize { sum32[i] += Float(cb0Ptr[i]) }

            // Add CB1-15 embeddings
            for (cbIdx, token) in cpTokens.enumerated() {
                let mceEmb = ensureNCHW(
                    try multiCodeEmbedder.embed(codebookIdx: cbIdx, tokenId: Int(token)),
                    channels: hiddenSize)
                let ptr = mceEmb.dataPointer.assumingMemoryBound(to: Float16.self)
                for i in 0..<hiddenSize { sum32[i] += Float(ptr[i]) }
            }
            embedSec += CFAbsoluteTimeGetCurrent() - tEmb0

            // Add tts_pad (FP32 from npy)
            if ttsPadEmbed.dataType == .float32 {
                let padPtr = ttsPadEmbed.dataPointer.assumingMemoryBound(to: Float.self)
                for i in 0..<hiddenSize { sum32[i] += padPtr[i] }
            } else {
                let padPtr = ttsPadEmbed.dataPointer.assumingMemoryBound(to: Float16.self)
                for i in 0..<hiddenSize { sum32[i] += Float(padPtr[i]) }
            }

            // Cast to FP16 for model input
            let stepInput = try MLMultiArray(shape: [1, NSNumber(value: hiddenSize), 1, 1], dataType: .float16)
            let outPtr = stepInput.dataPointer.assumingMemoryBound(to: Float16.self)
            for i in 0..<hiddenSize { outPtr[i] = Float16(sum32[i]) }

            // CodeDecoder forward
            let tCd = CFAbsoluteTimeGetCurrent()
            (lastLogits, _) = try codeDecoder.forward(embedArray: stepInput)
            cdDecodeSec += CFAbsoluteTimeGetCurrent() - tCd
            cdDecodeCalls += 1
            lastHidden = codeDecoder.lastHiddenState!

            // Sample CB0 — suppress control tokens [2048, 3072) except EOS
            let eosLogit = lastLogits[codecEos]
            for i in 2048..<codecVocabSize { if i != codecEos { lastLogits[i] = -1e9 } }
            if step < 2 { lastLogits[codecEos] = -1e9 }  // min_new_tokens=2
            else { lastLogits[codecEos] = eosLogit }

            // Repetition penalty
            for t in Set(generatedCB0) {
                let idx = Int(t)
                if lastLogits[idx] > 0 { lastLogits[idx] /= repetitionPenalty }
                else { lastLogits[idx] *= repetitionPenalty }
            }

            nextToken = TTSSampler.sample(
                logits: lastLogits, temperature: temperature, topK: topK)
            generatedCB0.append(nextToken)

            if nextToken == Int32(codecEos) { break }
            allCodebooks[0].append(nextToken)

            // MultiCodeDecoder: predict CB1-15
            let tMcd = CFAbsoluteTimeGetCurrent()
            cpTokens = try multiCodeDecoder.predict(
                hiddenState: lastHidden, cb0Token: nextToken,
                codeEmbedder: codeEmbedder, multiCodeEmbedder: multiCodeEmbedder,
                temperature: temperature, topK: topK)
            mcdSec += CFAbsoluteTimeGetCurrent() - tMcd
            mcdPredictCalls += 1
            for (i, token) in cpTokens.enumerated() {
                allCodebooks[i + 1].append(token)
            }
        }
        tDecodeEnd = CFAbsoluteTimeGetCurrent()

        // SpeechDecoder: all codes → audio
        guard !allCodebooks[0].isEmpty else { return [] }
        var audio = try speechDecoder.decode(codes: allCodebooks)
        tVocoderEnd = CFAbsoluteTimeGetCurrent()

        // Normalize amplitude
        let peak = audio.map { abs($0) }.max() ?? 0
        if peak > 0.001 {
            let gain = min(0.9 / peak, 10.0)
            for i in 0..<audio.count { audio[i] *= gain }
        }

        if benchOn {
            let total = tVocoderEnd - tStart
            let audioSec = Double(audio.count) / 24000.0
            let rtfx = audioSec / max(total, 1e-9)
            let mcdPerCallMs = mcdPredictCalls > 0 ? mcdSec / Double(mcdPredictCalls) * 1000 : 0
            let cdDecodePerCallMs = cdDecodeCalls > 0 ? cdDecodeSec / Double(cdDecodeCalls) * 1000 : 0
            // MCD internally calls its CoreML model 16x per predict() (positions 0..15)
            let mcdInnerCalls = mcdPredictCalls * 16
            let mcdInnerPerCallMs = mcdInnerCalls > 0 ? mcdSec / Double(mcdInnerCalls) * 1000 : 0
            func f(_ x: Double) -> String { String(format: "%6.3f", x) }
            print("""

            [BENCH Qwen3TTSCoreML]
              text                = \"\(text)\"
              audio output        = \(f(audioSec))s
              wall total          = \(f(total))s   RTFx = \(String(format: "%.2f", rtfx))
              prompt build        = \(f(tPromptEnd - tStart))s
              CD prefill          = \(f(cdPrefillSec))s  (\(cdPrefillCalls) calls)
              CD decode           = \(f(cdDecodeSec))s  (\(cdDecodeCalls) calls, \(String(format: "%.1f", cdDecodePerCallMs))ms/call)
              MCD predict (outer) = \(f(mcdSec))s  (\(mcdPredictCalls) calls, \(String(format: "%.1f", mcdPerCallMs))ms/call)
              MCD inner CoreML    = \(mcdInnerCalls) calls, \(String(format: "%.1f", mcdInnerPerCallMs))ms/call
              embedder sum loop   = \(f(embedSec))s
              SpeechDecoder       = \(f(tVocoderEnd - tDecodeEnd))s
              accounted           = \(f(cdPrefillSec + cdDecodeSec + mcdSec + embedSec + (tVocoderEnd - tDecodeEnd) + (tPromptEnd - tStart)))s
            """)
        }

        return audio
    }

    public func unload() {
        codeDecoder = nil; multiCodeDecoder = nil; speechDecoder = nil
        textProjector = nil; codeEmbedder = nil; multiCodeEmbedder = nil
        tokenizer = nil
    }

    public enum TTSCoreMLError: Error {
        case modelNotLoaded
    }
}
#endif
