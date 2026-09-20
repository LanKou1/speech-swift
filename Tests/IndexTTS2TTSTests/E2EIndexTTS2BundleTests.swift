import AudioCommon
import Darwin
import Foundation
@testable import IndexTTS2TTS
import MLX
@testable import Qwen3ASR
import XCTest

final class E2EIndexTTS2BundleTests: XCTestCase {
    private static let defaultRoundtripText = "Hello world. This is a local voice cloning round trip test."
    private static let defaultQwenASRModelId = "aufklarer/Qwen3-ASR-0.6B-MLX-4bit"

    func testExpandedBundleLoadsAndReportsCurrentSynthesisStatus() async throws {
        let model = try await loadModel()

        XCTAssertEqual(model.manifest.modelKey, IndexTTS2TTSModel.modelKey)
        XCTAssertEqual(model.manifest.displayName, "IndexTTS2")
        XCTAssertEqual(model.manifest.sampleRateHz, 24_000)
        XCTAssertEqual(model.manifest.auxiliaryModels.map(\.sourceRepo), [
            "facebook/w2v-bert-2.0",
            "amphion/MaskGCT",
            "funasr/campplus",
            "nvidia/bigvgan_v2_22khz_80band_256x",
        ])
        XCTAssertGreaterThan(model.memoryFootprint, 4_000_000_000)
        XCTAssertNotNil(model.tokenizer)
        XCTAssertEqual(model.runtimeConfig?.outputSampleRate, 22_050)

        let inventory = try model.prepareRuntime()
        XCTAssertEqual(inventory.outputSampleRate, 22_050)
        XCTAssertEqual(inventory.gptTensorCount, 665)
        XCTAssertEqual(inventory.s2MelRuntimeTensorCount, 284)
        XCTAssertEqual(inventory.s2MelIgnoredTensorCount, 1_114)
        XCTAssertEqual(inventory.w2vBertTensorCount, 773)
        XCTAssertEqual(inventory.semanticCodecTensorCount, 239)
        XCTAssertEqual(inventory.campPlusTensorCount, 937)
        XCTAssertEqual(inventory.bigVGANTensorCount, 783)
        XCTAssertEqual(inventory.wavStatTensorCount, 2)
        XCTAssertEqual(inventory.speakerMatrixShape, [73, 192])
        XCTAssertEqual(inventory.emotionMatrixShape, [73, 1280])
        XCTAssertEqual(inventory.qwenEmotionTensorCount, 310)

        let referenceURL = try makeReferenceWAV()
        let conditioning = try model.prepareReferenceConditioning(referenceAudio: referenceURL)
        XCTAssertEqual(conditioning.speakerInputFeatureShape.count, 3)
        XCTAssertEqual(conditioning.speakerInputFeatureShape[0], 1)
        XCTAssertEqual(conditioning.speakerInputFeatureShape[2], 160)
        XCTAssertEqual(conditioning.speakerAttentionMaskShape, [
            1,
            conditioning.speakerInputFeatureShape[1],
        ])
        XCTAssertEqual(conditioning.speakerSemanticHiddenShape, [
            1,
            conditioning.speakerInputFeatureShape[1],
            1024,
        ])
        XCTAssertEqual(conditioning.speakerSemanticCodeShape, [
            1,
            conditioning.speakerInputFeatureShape[1],
        ])
        XCTAssertEqual(conditioning.speakerSemanticPromptShape, conditioning.speakerSemanticHiddenShape)
        XCTAssertEqual(conditioning.emotionInputFeatureShape, conditioning.speakerInputFeatureShape)
        XCTAssertEqual(conditioning.emotionAttentionMaskShape, conditioning.speakerAttentionMaskShape)
        XCTAssertEqual(conditioning.emotionSemanticHiddenShape, conditioning.speakerSemanticHiddenShape)
        XCTAssertEqual(conditioning.promptMelShape.count, 3)
        XCTAssertEqual(conditioning.promptMelShape[0], 1)
        XCTAssertEqual(conditioning.promptMelShape[1], 80)
        XCTAssertGreaterThan(conditioning.promptMelShape[2], 0)
        XCTAssertEqual(conditioning.promptConditionShape, [
            1,
            conditioning.promptMelShape[2],
            512,
        ])
        XCTAssertEqual(conditioning.styleEmbeddingShape, [1, 192])
        XCTAssertGreaterThan(conditioning.reference16kSampleCount, 16_000)
        XCTAssertGreaterThan(conditioning.reference22kSampleCount, 22_050)

        let semantic = try model.generateSemanticCodes(
            text: "HELLO WORLD",
            conditioning: conditioning,
            options: IndexTTS2SemanticGenerationOptions(maxSemanticTokens: 2))
        XCTAssertLessThanOrEqual(semantic.codeCount, 2)
        XCTAssertEqual(semantic.codeShape, [1, semantic.codeCount])
        XCTAssertEqual(semantic.conditioningLatentShape, [1, 32, 1280])
        XCTAssertTrue(semantic.codes.allSatisfy { (0..<8192).contains(Int($0)) })

        let audio = try model.synthesize(
            text: "HELLO WORLD",
            conditioning: conditioning,
            semanticOptions: IndexTTS2SemanticGenerationOptions(maxSemanticTokens: 2))
        XCTAssertGreaterThan(audio.count, 0)
        XCTAssertTrue(audio.allSatisfy(\.isFinite))
        XCTAssertLessThanOrEqual(audio.map(abs).max() ?? 0, 1.01)
    }

    func testFullSynthesisBenchmarkAndOptionalASRRoundtrip() async throws {
        let env = ProcessInfo.processInfo.environment
        let model = try await loadModel()
        let referenceURL = try benchmarkReferenceURL(env: env)
        let outputURL = benchmarkOutputURL(env: env)
        let text = env["INDEXTTS2_E2E_TEXT"] ?? Self.defaultRoundtripText
        let maxSemanticTokens = Int(env["INDEXTTS2_E2E_MAX_SEMANTIC_TOKENS"] ?? "") ?? 240
        let semanticOptions = Self.semanticOptions(env: env, maxSemanticTokens: maxSemanticTokens)
        let synthesisOptions = try Self.synthesisOptions(env: env)
        let emotionControl = try Self.emotionControl(env: env)

        let rssBeforeRuntime = Self.currentResidentMemoryBytes()
        let runtimeStart = CFAbsoluteTimeGetCurrent()
        _ = try model.prepareRuntime()
        let runtimeSec = CFAbsoluteTimeGetCurrent() - runtimeStart
        let rssAfterRuntime = Self.currentResidentMemoryBytes()

        let conditioningStart = CFAbsoluteTimeGetCurrent()
        let conditioning = try model.prepareReferenceConditioning(
            referenceAudio: referenceURL,
            emotionControl: emotionControl)
        let conditioningSec = CFAbsoluteTimeGetCurrent() - conditioningStart
        let rssAfterConditioning = Self.currentResidentMemoryBytes()

        if let seeds = Self.seedSweep(env: env), !seeds.isEmpty {
            for seed in seeds {
                var options = semanticOptions
                options.seed = seed
                let semantic = try model.generateSemanticCodes(
                    text: text,
                    conditioning: conditioning,
                    options: options)
                let prefix = semantic.codes.prefix(24).map(String.init).joined(separator: ",")
                print("[IndexTTS2SemanticSeed] seed=\(seed) generatedTokens=\(semantic.codeCount) prefix=\(prefix)")
            }
            return
        }

        let synthesisStart = CFAbsoluteTimeGetCurrent()
        let audio: [Float]
        if let suppliedCodes = Self.semanticCodes(env: env) {
            audio = try model.synthesize(
                text: text,
                conditioning: conditioning,
                semanticCodes: suppliedCodes,
                synthesisOptions: synthesisOptions)
            let prefix = suppliedCodes.prefix(24).map(String.init).joined(separator: ",")
            print("[IndexTTS2Semantic] suppliedTokens=\(suppliedCodes.count) prefix=\(prefix)")
        } else {
            let semantic = try model.generateSemanticCodes(
                text: text,
                conditioning: conditioning,
                options: semanticOptions)
            let prefix = semantic.codes.prefix(24).map(String.init).joined(separator: ",")
            print("[IndexTTS2Semantic] generatedTokens=\(semantic.codeCount) prefix=\(prefix)")
            if env["INDEXTTS2_E2E_SEMANTIC_ONLY"] == "1" {
                return
            }
            audio = try model.synthesize(
                text: text,
                conditioning: conditioning,
                semantic: semantic,
                synthesisOptions: synthesisOptions)
        }
        let synthesisSec = CFAbsoluteTimeGetCurrent() - synthesisStart
        let rssAfterSynthesis = Self.currentResidentMemoryBytes()

        let audioSec = Double(audio.count) / Double(model.sampleRate)
        let synthesisRTF = synthesisSec / max(audioSec, 1e-6)
        let pipelineSec = conditioningSec + synthesisSec
        let pipelineRTF = pipelineSec / max(audioSec, 1e-6)
        let rms = Self.rms(audio)
        try WAVWriter.write(samples: audio, sampleRate: model.sampleRate, to: outputURL)

        XCTAssertGreaterThan(audio.count, 0)
        XCTAssertGreaterThan(audioSec, 0.1)
        XCTAssertTrue(audio.allSatisfy(\.isFinite))
        XCTAssertGreaterThan(rms, 1e-4)

        print(String(format:
            "[IndexTTS2Benchmark] output=%@ reference=%@ samples=%d sampleRate=%d audioSec=%.3f runtimeSec=%.3f conditioningSec=%.3f synthesisSec=%.3f synthesisRTF=%.3f pipelineRTF=%.3f rms=%.6f modelFootprintMB=%.1f rssBeforeRuntimeMB=%.1f rssAfterRuntimeMB=%.1f rssAfterConditioningMB=%.1f rssAfterSynthesisMB=%.1f rssRuntimeDeltaMB=%.1f rssPeakDeltaMB=%.1f",
            outputURL.path,
            referenceURL.path,
            audio.count,
            model.sampleRate,
            audioSec,
            runtimeSec,
            conditioningSec,
            synthesisSec,
            synthesisRTF,
            pipelineRTF,
            rms,
            Self.megabytes(UInt64(model.memoryFootprint)),
            Self.megabytes(rssBeforeRuntime),
            Self.megabytes(rssAfterRuntime),
            Self.megabytes(rssAfterConditioning),
            Self.megabytes(rssAfterSynthesis),
            Self.megabytes(rssAfterRuntime > rssBeforeRuntime ? rssAfterRuntime - rssBeforeRuntime : 0),
            Self.megabytes(max(rssAfterRuntime, rssAfterConditioning, rssAfterSynthesis) > rssBeforeRuntime
                ? max(rssAfterRuntime, rssAfterConditioning, rssAfterSynthesis) - rssBeforeRuntime
                : 0)))

        guard env["INDEXTTS2_E2E_ROUNDTRIP"] == "1" else {
            print("[IndexTTS2Roundtrip] skipped; set INDEXTTS2_E2E_ROUNDTRIP=1 to load ASR and measure WER")
            return
        }

        let asrModelId = env["INDEXTTS2_E2E_ASR_MODEL"] ?? Self.defaultQwenASRModelId
        let asr = try await Qwen3ASRModel.fromPretrained(modelId: asrModelId)
        let asrStart = CFAbsoluteTimeGetCurrent()
        let language = env["INDEXTTS2_E2E_LANGUAGE"] ?? "english"
        let transcript = asr.transcribe(audio: audio, sampleRate: model.sampleRate, language: language)
        let asrSec = CFAbsoluteTimeGetCurrent() - asrStart
        let asrRTF = asrSec / max(audioSec, 1e-6)
        // ASR writes numbers as digits, so score both sides after the same
        // number reading the synthesis used. CJK references carry no word
        // boundaries; score them per character.
        let scoredReference = IndexTTS2TextNormalizer.normalize(text)
        let scoredHypothesis = IndexTTS2TextNormalizer.normalize(transcript)
        let usesCharacters = text.unicodeScalars.contains(where: Self.isCJK)
        let wer = usesCharacters
            ? Self.characterErrorRate(reference: scoredReference, hypothesis: scoredHypothesis)
            : Self.wordErrorRate(reference: scoredReference, hypothesis: scoredHypothesis)
        let maxWER = Double(env["INDEXTTS2_E2E_MAX_WER"] ?? "") ?? 0.25

        print(String(format:
            "[IndexTTS2Roundtrip] asrModel=%@ language=%@ metric=%@ transcript=\"%@\" wer=%.3f asrSec=%.3f asrRTF=%.3f maxWER=%.3f",
            asrModelId,
            language,
            usesCharacters ? "cer" : "wer",
            transcript,
            wer,
            asrSec,
            asrRTF,
            maxWER))

        XCTAssertFalse(Self.normalizedWords(transcript).isEmpty, "ASR roundtrip transcript should not be empty")
        XCTAssertLessThanOrEqual(wer, maxWER, "ASR roundtrip WER exceeded threshold")
    }

    func testTokenizerMatchesUpstreamTokenIDs() async throws {
        let tokenizer = try await loadTokenizer()

        // Golden ids from the upstream text front end (`char_rep_map` +
        // `tokenize_by_CJK_char`) and SentencePiece on the published `bpe.model`.
        let cases: [(text: String, ids: [Int])] = [
            ("你好世界", [10201, 208, 10201, 1260, 10201, 22, 10201, 3755]),
            ("你好，这是一个中文合成测试。",
             [10201, 208, 10201, 1260, 10202, 10201, 5935, 10201, 2474, 10201, 7, 10201, 34, 10201, 36,
              10201, 2398, 10201, 680, 10201, 2043, 10201, 3110, 10201, 5551, 10203]),
            ("你好世界是 hello world 的中文",
             [10201, 208, 10201, 1260, 10201, 22, 10201, 3755, 10201, 2474, 11122, 10394, 10201, 3880,
              10201, 36, 10201, 2398]),
            ("你好！吗？", [10201, 208, 10201, 1260, 10201, 10481, 10201, 694, 10219]),
            ("Hello \"world\" (ok): done.",
             [11122, 10201, 10206, 10603, 10391, 10438, 10258, 10206, 10201, 10206, 11412, 10206,
              10209, 10467, 10216]),
            ("你好🙂", [10201, 208, 10201, 1260, 10201, 2]),
            ("A2024B", [10210, 10270, 10456, 10380, 10361, 10462]),
            ("今天是2024年",
             [10201, 124, 10201, 1221, 10201, 2474, 10201, 83, 10201, 6500, 10201, 83, 10201, 1017, 10201, 1698]),
            ("今天是2024年3月5日，温度是25.5度。",
             [10201, 124, 10201, 1221, 10201, 2474, 10201, 83, 10201, 6500, 10201, 83, 10201, 1017, 10201, 1698,
              10201, 12, 10201, 2544, 10201, 90, 10201, 2437, 10202, 10201, 3210, 10201, 1726, 10201, 2474,
              10201, 83, 10201, 570, 10201, 90, 10201, 3393, 10201, 90, 10201, 1726, 10203]),
            ("The meeting is at 10:30 on the 21st.",
             [10204, 11762, 10218, 10244, 10472, 10561, 10226, 10204, 10380, 10315, 10216]),
            ("It costs $3.50, about 50% off.",
             [10215, 10985, 10208, 10321, 10681, 10601, 10201, 10336, 11394, 10208, 10209, 10251, 10601, 10532,
              10375, 10216]),
        ]
        for (text, ids) in cases {
            XCTAssertEqual(try tokenizer.encode(text), ids, "token ids for \(text)")
        }
    }

    private func loadTokenizer() async throws -> IndexTTS2Tokenizer {
        let env = ProcessInfo.processInfo.environment
        if let path = env["INDEXTTS2_E2E_BPE_MODEL"], !path.isEmpty {
            let url = URL(fileURLWithPath: path)
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw XCTSkip("INDEXTTS2_E2E_BPE_MODEL not found at \(url.path)")
            }
            return try IndexTTS2Tokenizer(modelURL: url)
        }
        let model = try await loadModel()
        return try XCTUnwrap(model.tokenizer, "bundle tokenizer failed to initialize")
    }

    private static let defaultLongText =
        "The quick brown fox jumps over the lazy dog near the river bank. "
        + "Every morning the baker opens the shop before sunrise and sets out fresh bread. "
        + "Later in the day the children walk home from school along the same road. "
        + "By evening the whole street smells of coffee and rain."

    func testLongTextSynthesisSplitsIntoSegments() async throws {
        let model = try await loadModel()
        let env = ProcessInfo.processInfo.environment
        let referenceURL = try benchmarkReferenceURL(env: env)
        let text = env["INDEXTTS2_E2E_LONG_TEXT"] ?? Self.defaultLongText
        let options = try IndexTTS2SynthesisOptions(
            maxInternalPauseDuration: Float(env["INDEXTTS2_E2E_MAX_PAUSE"] ?? ""),
            maxTextTokensPerSegment: Int(env["INDEXTTS2_E2E_SEGMENT_TOKENS"] ?? "") ?? 30)
        let tokenizer = try XCTUnwrap(model.tokenizer)
        let segments = IndexTTS2TextSegmenter.split(
            try tokenizer.tokenize(text), maxTokens: options.maxTextTokensPerSegment)
        XCTAssertGreaterThanOrEqual(segments.count, 3, "long text should split into several segments")

        _ = try model.prepareRuntime()
        let conditioning = try model.prepareReferenceConditioning(referenceAudio: referenceURL)
        let start = CFAbsoluteTimeGetCurrent()
        let audio = try model.synthesize(text: text, conditioning: conditioning, synthesisOptions: options)
        let synthesisSec = CFAbsoluteTimeGetCurrent() - start
        let audioSec = Double(audio.count) / Double(model.sampleRate)
        XCTAssertGreaterThan(audioSec, 8)
        XCTAssertTrue(audio.allSatisfy(\.isFinite))
        if let output = env["INDEXTTS2_E2E_OUTPUT"], !output.isEmpty {
            try WAVWriter.write(samples: audio, sampleRate: model.sampleRate, to: URL(fileURLWithPath: output))
        }
        print(String(format: "[IndexTTS2LongText] segments=%d audioSec=%.2f synthesisSec=%.1f rtf=%.2f",
                     segments.count, audioSec, synthesisSec, synthesisSec / max(audioSec, 1e-6)))

        guard env["INDEXTTS2_E2E_ROUNDTRIP"] == "1" else { return }
        Memory.clearCache()
        let asr = try await Qwen3ASRModel.fromPretrained(
            modelId: env["INDEXTTS2_E2E_ASR_MODEL"] ?? Self.defaultQwenASRModelId)
        let language = env["INDEXTTS2_E2E_LANGUAGE"] ?? "english"
        let transcript = asr.transcribe(audio: audio, sampleRate: model.sampleRate, language: language)
        let usesCharacters = text.unicodeScalars.contains(where: Self.isCJK)
        let wer = usesCharacters
            ? Self.characterErrorRate(reference: text, hypothesis: transcript)
            : Self.wordErrorRate(reference: text, hypothesis: transcript)
        print(String(format: "[IndexTTS2LongTextRoundtrip] transcript=\"%@\" wer=%.3f", transcript, wer))
        XCTAssertLessThanOrEqual(wer, Double(env["INDEXTTS2_E2E_MAX_WER"] ?? "") ?? 0.25)
    }

    private func loadModel() async throws -> IndexTTS2TTSModel {
        let env = ProcessInfo.processInfo.environment
        if let bundlePath = env["INDEXTTS2_E2E_BUNDLE"], !bundlePath.isEmpty {
            let bundle = URL(fileURLWithPath: bundlePath, isDirectory: true)
            guard FileManager.default.fileExists(atPath: bundle.path) else {
                throw XCTSkip("IndexTTS2 E2E bundle not found at \(bundle.path)")
            }
            return try await IndexTTS2TTSModel.fromBundle(bundle)
        }

        guard env["INDEXTTS2_E2E_DOWNLOAD"] == "1" else {
            throw XCTSkip("Set INDEXTTS2_E2E_BUNDLE or INDEXTTS2_E2E_DOWNLOAD=1 to run IndexTTS2 bundle E2E")
        }

        do {
            return try await IndexTTS2TTSModel.fromPretrained()
        } catch {
            throw XCTSkip("IndexTTS2 published bundle unavailable: \(error)")
        }
    }

    private func makeReferenceWAV() throws -> URL {
        let sampleRate = 16_000
        let sampleCount = sampleRate * 2
        let samples = (0..<sampleCount).map { i -> Float in
            let t = Float(i) / Float(sampleRate)
            return 0.12 * sin(2.0 * Float.pi * 220.0 * t)
        }
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("indextts2-reference-\(UUID().uuidString).wav")
        try WAVWriter.write(samples: samples, sampleRate: sampleRate, to: url)
        return url
    }

    private func benchmarkReferenceURL(env: [String: String]) throws -> URL {
        if let path = env["INDEXTTS2_E2E_REFERENCE"], !path.isEmpty {
            let url = URL(fileURLWithPath: path)
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw XCTSkip("INDEXTTS2_E2E_REFERENCE not found at \(url.path)")
            }
            return url
        }

        let bundled = URL(fileURLWithPath: "Tests/OmnilingualASRTests/Resources/fleurs_en.wav")
        if FileManager.default.fileExists(atPath: bundled.path) {
            return bundled
        }
        throw XCTSkip("Set INDEXTTS2_E2E_REFERENCE to a real reference WAV for the full IndexTTS2 benchmark")
    }

    private func benchmarkOutputURL(env: [String: String]) -> URL {
        if let path = env["INDEXTTS2_E2E_OUTPUT"], !path.isEmpty {
            return URL(fileURLWithPath: path)
        }
        return FileManager.default.temporaryDirectory
            .appendingPathComponent("indextts2-e2e-\(UUID().uuidString).wav")
    }

    private static func semanticCodes(env: [String: String]) -> [Int32]? {
        guard let raw = env["INDEXTTS2_E2E_SEMANTIC_CODES"], !raw.isEmpty else {
            return nil
        }
        let codes = raw
            .split { $0 == "," || $0 == " " || $0 == "\n" || $0 == "\t" }
            .compactMap { Int32($0) }
        return codes.isEmpty ? nil : codes
    }

    private static func semanticOptions(
        env: [String: String],
        maxSemanticTokens: Int
    ) -> IndexTTS2SemanticGenerationOptions {
        IndexTTS2SemanticGenerationOptions(
            maxSemanticTokens: maxSemanticTokens,
            greedy: env["INDEXTTS2_E2E_GREEDY"] == "1",
            temperature: Float(env["INDEXTTS2_E2E_TEMPERATURE"] ?? "") ?? 0.8,
            topK: Int(env["INDEXTTS2_E2E_TOP_K"] ?? "") ?? 30,
            topP: Float(env["INDEXTTS2_E2E_TOP_P"] ?? "") ?? 0.8,
            repetitionPenalty: Float(env["INDEXTTS2_E2E_REPETITION_PENALTY"] ?? "") ?? 10.0,
            seed: UInt64(env["INDEXTTS2_E2E_SEED"] ?? "") ?? 11,
            beamWidth: Int(env["INDEXTTS2_E2E_BEAMS"] ?? "") ?? 3,
            lengthPenalty: Float(env["INDEXTTS2_E2E_LENGTH_PENALTY"] ?? "") ?? 0.0)
    }

    private static func emotionControl(env: [String: String]) throws -> IndexTTS2EmotionControl? {
        guard let raw = env["INDEXTTS2_E2E_EMOTION"]?.trimmingCharacters(in: .whitespacesAndNewlines),
              !raw.isEmpty else {
            return nil
        }
        let weight = Float(env["INDEXTTS2_E2E_EMOTION_WEIGHT"] ?? "") ?? 1.0
        if let preset = IndexTTS2EmotionPreset(named: raw) {
            return try IndexTTS2EmotionControl(preset: preset, weight: weight)
        }
        var text = raw
        if text.hasPrefix("[") && text.hasSuffix("]") {
            text.removeFirst()
            text.removeLast()
        }
        let parts = text.split(separator: ",", omittingEmptySubsequences: false)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
        let vector = try parts.map { part -> Float in
            guard let value = Float(part) else {
                throw XCTSkip("INDEXTTS2_E2E_EMOTION must be a preset or 8-value numeric vector")
            }
            return value
        }
        return try IndexTTS2EmotionControl(vector: vector, weight: weight)
    }

    private static func synthesisOptions(env: [String: String]) throws -> IndexTTS2SynthesisOptions {
        try IndexTTS2SynthesisOptions(
            speakingRate: Float(env["INDEXTTS2_E2E_SPEAKING_RATE"] ?? "") ?? 1.0,
            maxInternalPauseDuration: Float(env["INDEXTTS2_E2E_MAX_PAUSE"] ?? ""))
    }

    private static func seedSweep(env: [String: String]) -> [UInt64]? {
        guard let raw = env["INDEXTTS2_E2E_SEED_SWEEP"], !raw.isEmpty else {
            return nil
        }
        let range = raw.split(separator: "-")
        if range.count == 2,
            let start = UInt64(range[0]),
            let end = UInt64(range[1]),
            start <= end {
            return Array(start...end)
        }
        let parts = raw.split { scalar in
            scalar == "," || scalar == " " || scalar == "\n" || scalar == "\t"
        }
        return parts.compactMap { UInt64($0) }
    }

    private static func rms(_ samples: [Float]) -> Double {
        guard !samples.isEmpty else { return 0 }
        let sumSquares = samples.reduce(0.0) { $0 + Double($1) * Double($1) }
        return (sumSquares / Double(samples.count)).squareRoot()
    }

    private static func currentResidentMemoryBytes() -> UInt64 {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
        let result = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { rebound in
                task_info(
                    mach_task_self_,
                    task_flavor_t(MACH_TASK_BASIC_INFO),
                    rebound,
                    &count)
            }
        }
        guard result == KERN_SUCCESS else { return 0 }
        return UInt64(info.resident_size)
    }

    private static func megabytes(_ bytes: UInt64) -> Double {
        Double(bytes) / 1_048_576.0
    }

    private static func wordErrorRate(reference: String, hypothesis: String) -> Double {
        editRate(reference: normalizedWords(reference), hypothesis: normalizedWords(hypothesis))
    }

    private static func characterErrorRate(reference: String, hypothesis: String) -> Double {
        editRate(reference: normalizedCharacters(reference), hypothesis: normalizedCharacters(hypothesis))
    }

    private static func editRate(reference ref: [String], hypothesis hyp: [String]) -> Double {
        guard !ref.isEmpty else { return hyp.isEmpty ? 0 : 1 }

        var previous = Array(0...hyp.count)
        for (i, refWord) in ref.enumerated() {
            var current = [i + 1] + Array(repeating: 0, count: hyp.count)
            for (j, hypWord) in hyp.enumerated() {
                let substitution = previous[j] + (refWord == hypWord ? 0 : 1)
                let insertion = current[j] + 1
                let deletion = previous[j + 1] + 1
                current[j + 1] = min(substitution, insertion, deletion)
            }
            previous = current
        }
        return Double(previous[hyp.count]) / Double(ref.count)
    }

    private static func normalizedCharacters(_ text: String) -> [String] {
        normalizedWords(text).flatMap { $0.map(String.init) }
    }

    private static func isCJK(_ scalar: Unicode.Scalar) -> Bool {
        (0x2E80...0x9FFF).contains(scalar.value)
            || (0xF900...0xFAFF).contains(scalar.value)
            || (0x20000...0x2FFFF).contains(scalar.value)
    }

    private static func normalizedWords(_ text: String) -> [String] {
        text.lowercased()
            .unicodeScalars
            .map { CharacterSet.alphanumerics.contains($0) ? Character($0) : " " }
            .reduce(into: "") { $0.append($1) }
            .split(whereSeparator: { $0.isWhitespace })
            .map(String.init)
    }
}
