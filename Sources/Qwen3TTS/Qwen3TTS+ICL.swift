import Foundation
import MLX
import MLXNN
import MLXFast
import MLXCommon
import AudioCommon
import os

// MARK: - ICL Voice Cloning

extension Qwen3TTSModel {

    /// Load a Qwen3-TTS model together with the SpeechTokenizerEncoder for ICL voice cloning.
    ///
    /// The encoder is used to convert reference audio into codec tokens. Weights come from
    /// the same safetensors file (encoder.* prefixes).
    public static func fromPretrainedWithEncoder(
        // Default to the 1.7B-bf16 (the production model Studio ships) — NOT the
        // decommissioned 0.6B int4. The int4 default previously masked the 1.7B
        // speaker-encoder bug (2048-dim fc + PyTorch conv layout), since the e2e
        // tests only ever exercised the 0.6B path.
        modelId: String = "aufklarer/Qwen3-TTS-12Hz-1.7B-Base-MLX-bf16",
        tokenizerModelId: String = "Qwen/Qwen3-TTS-Tokenizer-12Hz",
        cacheDir: URL? = nil,
        tokenizerCacheDir: URL? = nil,
        offlineMode: Bool = false,
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> (Qwen3TTSModel, SpeechTokenizerEncoder) {
        let tts = try await Qwen3TTSModel.fromPretrained(
            modelId: modelId,
            tokenizerModelId: tokenizerModelId,
            cacheDir: cacheDir,
            tokenizerCacheDir: tokenizerCacheDir,
            offlineMode: offlineMode,
            progressHandler: progressHandler)

        // The codec encoder (Mimi) lives in the speech-tokenizer bundle, NOT the
        // talker bundle — the talker bundle has no `encoder.*`/`decoder.*` keys.
        // `fromPretrained` already downloaded the tokenizer (same dir the decoder
        // loads from), so just resolve its cache dir.
        let tokenizerDir = try tokenizerCacheDir
            ?? HuggingFaceDownloader.getCacheDirectory(for: tokenizerModelId)

        let encoderConfig = tts.config.speechTokenizerDecoder
        let encoder = SpeechTokenizerEncoder(config: encoderConfig)
        try TTSWeightLoader.loadSpeechTokenizerEncoderWeights(into: encoder, from: tokenizerDir)

        return (tts, encoder)
    }

    /// Synthesize speech using In-Context Learning (ICL) voice cloning.
    ///
    /// Unlike `synthesizeWithVoiceClone()` (x-vector only), ICL encodes the reference audio
    /// into codec tokens and prepends them with their transcript into the autoregressive
    /// context. This produces correct EOS and higher voice fidelity.
    ///
    /// The reference codec (from the encoder) is supplied as in-context conditioning, so
    /// the model generates the TARGET codec only. For decoding, the reference codec is
    /// prepended (the causal Mimi decoder needs left context) and then cut back off by the
    /// exact `refFrames / totalFrames` audio ratio — leaving clean target audio. Pass
    /// `trimReference: false` to receive the full decoded waveform (reference + target).
    ///
    /// - Parameters:
    ///   - text: Target text to synthesize
    ///   - referenceAudio: Raw PCM samples of the reference recording
    ///   - referenceSampleRate: Sample rate of referenceAudio (resampled to 24kHz internally)
    ///   - referenceText: Exact transcript of the reference recording
    ///   - language: Language hint (e.g. "english", "german")
    ///   - sampling: Sampling config
    ///   - codecEncoder: SpeechTokenizerEncoder from fromPretrainedWithEncoder()
    ///   - trimReference: When `true` (default), prepend the reference codec for the decode
    ///                    and cut the reference portion (`refFrames / totalFrames` of the
    ///                    audio) off the front, returning only the target audio. Set `false` to
    ///                    receive the raw decoded waveform.
    public func synthesizeWithVoiceCloneICL(
        text: String,
        referenceAudio: [Float],
        referenceSampleRate: Int = 24000,
        referenceText: String,
        language: String = "auto",
        sampling: SamplingConfig = .default,
        codecEncoder: SpeechTokenizerEncoder,
        trimReference: Bool = true
    ) -> [Float] {
        synthesizeWithVoiceCloneICL(
            text: text, referenceAudio: referenceAudio,
            referenceSampleRate: referenceSampleRate, referenceText: referenceText,
            language: language, sampling: sampling, codecEncoder: codecEncoder,
            trimReference: trimReference, checkCancellation: {})
    }

    /// Cancellation-aware ICL cloning. Checks cancellation between MLX stages
    /// and within autoregressive generation; an executing GPU kernel may finish.
    public func synthesizeWithVoiceCloneICLCancellable(
        text: String,
        referenceAudio: [Float],
        referenceSampleRate: Int = 24000,
        referenceText: String,
        language: String = "auto",
        sampling: SamplingConfig = .default,
        codecEncoder: SpeechTokenizerEncoder,
        trimReference: Bool = true
    ) throws -> [Float] {
        try synthesizeWithVoiceCloneICL(
            text: text, referenceAudio: referenceAudio,
            referenceSampleRate: referenceSampleRate, referenceText: referenceText,
            language: language, sampling: sampling, codecEncoder: codecEncoder,
            trimReference: trimReference,
            checkCancellation: { try Task.checkCancellation() })
    }

    private func synthesizeWithVoiceCloneICL(
        text: String,
        referenceAudio: [Float],
        referenceSampleRate: Int = 24000,
        referenceText: String,
        language: String = "auto",
        sampling: SamplingConfig = .default,
        codecEncoder: SpeechTokenizerEncoder,
        trimReference: Bool = true,
        checkCancellation: () throws -> Void
    ) rethrows -> [Float] {
        try checkCancellation()
        guard let tokenizer = tokenizer else {
            fatalError("Tokenizer not loaded")
        }
        // "auto" (matches the QwenLM reference + mlx-audio default) skips the
        // language-id token and switches the codec prefix to the codec_nothink
        // branch. Any other value must resolve to a known language id.
        let langId: Int?
        let normalized = language.lowercased()
        if normalized == "auto" || normalized.isEmpty {
            langId = nil
        } else if let id = CodecTokens.languageId(for: language) {
            langId = id
        } else {
            AudioLog.inference.warning(
                "Unknown language '\(language, privacy: .private)', falling back to auto")
            return try synthesizeWithVoiceCloneICL(
                text: text, referenceAudio: referenceAudio,
                referenceSampleRate: referenceSampleRate,
                referenceText: referenceText, language: "auto",
                sampling: sampling, codecEncoder: codecEncoder,
                trimReference: trimReference, checkCancellation: checkCancellation)
        }

        let t0 = CFAbsoluteTimeGetCurrent()

        // Step 1+2: Encode reference audio → codec tokens [1, 16, T_ref] (cached per reference)
        let refCodes: MLXArray
        if let cached = referenceAudioCache.codecRefCodes(for: referenceAudio, sampleRate: referenceSampleRate) {
            refCodes = cached
            AudioLog.inference.debug(
                "ICL: codec tokens cache hit (\(cached.dim(2), privacy: .public) frames)")
        } else {
            let audio24k = referenceSampleRate == 24000
                ? referenceAudio
                : AudioFileLoader.resample(referenceAudio, from: referenceSampleRate, to: 24000)
            try checkCancellation()
            let codes = codecEncoder.encode(samples: audio24k)
            eval(codes)
            try checkCancellation()
            referenceAudioCache.storeCodecRefCodes(codes, audio: referenceAudio, sampleRate: referenceSampleRate)
            refCodes = codes
            AudioLog.inference.debug(
                "ICL: encoded \(audio24k.count, privacy: .public) samples → \(codes.dim(2), privacy: .public) codec frames")
        }

        try checkCancellation()

        // Step 3: Extract speaker embedding (ICL still uses x-vector for speaker conditioning; cached)
        let speakerEmbed: MLXArray
        if let cached = referenceAudioCache.speakerEmbed(for: referenceAudio, sampleRate: referenceSampleRate) {
            speakerEmbed = cached
        } else {
            let mels = SpeakerMel.compute(audio: referenceAudio, sampleRate: referenceSampleRate)
            try checkCancellation()
            let embed = speakerEncoder(mels)  // [1, 1024]
            eval(embed)
            try checkCancellation()
            referenceAudioCache.storeSpeakerEmbed(embed, audio: referenceAudio, sampleRate: referenceSampleRate)
            speakerEmbed = embed
        }

        try checkCancellation()

        // Step 4: Build ICL prefill embeddings
        let (prefillEmbeds, trailingTextHidden, ttsPadEmbed) = buildICLPrefillEmbeddings(
            refCodes: refCodes,
            referenceText: referenceText,
            targetText: text,
            language: language,
            languageId: langId,
            speakerEmbed: speakerEmbed,
            tokenizer: tokenizer)

        eval(prefillEmbeds, trailingTextHidden, ttsPadEmbed)
        try checkCancellation()
        let t1 = CFAbsoluteTimeGetCurrent()

        // Step 5: Autoregressive generation. Auto-bump repetition_penalty to
        // >= 1.5 ONLY when caller is sampling (T > 0). Under sampling the
        // bump matches mlx-audio's recipe and prevents codec-token repetition
        // artifacts. Under greedy the bump is a degeneration trap (argmax
        // flips on Metal logit jitter) — leave the caller's value as-is.
        var iclSampling = sampling
        if iclSampling.temperature > 0 && iclSampling.repetitionPenalty < 1.5 {
            iclSampling.repetitionPenalty = 1.5
        }
        // Cap maxTokens so an under-EOS runaway can't exhaust GPU memory. The
        // model generates the TARGET codec only (the reference is provided as
        // context, not regenerated), so the budget is purely a per-target-text
        // allowance: 8 codec frames per BPE token is a loose upper bound on
        // natural speech rate (~12.5 fps), with a floor for very short lines.
        let targetTokenCount = tokenizer.encode(text).count
        let textDerivedCap = max(96, targetTokenCount * 8)
        iclSampling.maxTokens = min(iclSampling.maxTokens, textDerivedCap)
        let (allCodebooks, numFrames) = try generateWithCodePredictor(
            prefillEmbeds: prefillEmbeds,
            trailingTextHidden: trailingTextHidden,
            ttsPadEmbed: ttsPadEmbed,
            sampling: iclSampling,
            checkCancellation: checkCancellation)

        eval(allCodebooks)
        try checkCancellation()
        let t2 = CFAbsoluteTimeGetCurrent()

        guard numFrames > 0 else {
            AudioLog.inference.warning("ICL generation produced no tokens")
            return []
        }

        // Decode with the reference codec as context, retaining Python's original
        // 300-frame chunk boundaries and 25-frame left context. Skip only decoder
        // chunks whose waveform would be removed by the reference-prefix trim.
        let refFrames = refCodes.dim(2)
        let codesForDecode = trimReference
            ? concatenated([refCodes, allCodebooks], axis: 2)   // [1, 16, refFrames + numFrames]
            : allCodebooks
        let totalFrames = codesForDecode.dim(2)
        AudioLog.inference.debug(
            "ICL: decoding \(numFrames, privacy: .public) target frames (+ \(trimReference ? refFrames : 0, privacy: .public) ref ctx) → \(totalFrames, privacy: .public) frames")
        try checkCancellation()
        let trimmedWaveform = try codecDecoder.decode(
            codes: codesForDecode, chunkSize: 300, leftContext: 25,
            startFrame: trimReference ? refFrames : 0,
            checkCancellation: checkCancellation)
        try checkCancellation()
        let t3 = CFAbsoluteTimeGetCurrent()

        let audioDur = Double(trimmedWaveform.count) / 24000.0
        let encTime = String(format: "%.3f", t1-t0)
        let genTime = String(format: "%.3f", t2-t1)
        let msPerStep = String(format: "%.0f", (t2-t1)/Double(max(numFrames, 1))*1000)
        let decTime = String(format: "%.3f", t3-t2)
        let totTime = String(format: "%.3f", t3-t0)
        let audDur = String(format: "%.2f", audioDur)
        let rtf = String(format: "%.2f", (t3-t0)/max(audioDur, 0.001))
        AudioLog.inference.info(
            "ICL timing: encode=\(encTime, privacy: .public)s | generate=\(genTime, privacy: .public)s (\(numFrames, privacy: .public) steps, \(msPerStep, privacy: .public)ms/step) | decode=\(decTime, privacy: .public)s | total=\(totTime, privacy: .public)s | audio=\(audDur, privacy: .public)s | RTF=\(rtf, privacy: .public)")

        try checkCancellation()
        return trimmedWaveform
    }

    // MARK: - ICL Prefill Construction

    /// Build ICL prefill embeddings following the Python mlx-audio reference.
    ///
    /// Layout:
    /// ```
    /// [role_embed]                                    ← <|im_start|>assistant\n
    /// [tts_pad...tts_bos + codec_prefix]              ← codec prefix overlay
    /// [ref_text + target_text + tts_eos + codec_pad]  ← text overlay
    /// [codec_bos + ref_codec_embeds]                  ← codec ICL context
    /// ```
    private func buildICLPrefillEmbeddings(
        refCodes: MLXArray,         // [1, 16, T_ref]
        referenceText: String,
        targetText: String,
        language: String,
        languageId: Int?,
        speakerEmbed: MLXArray,     // [1, 1024]
        tokenizer: Qwen3Tokenizer
    ) -> (prefillEmbeds: MLXArray, trailingTextHidden: MLXArray, ttsPadEmbed: MLXArray) {
        let hiddenSize = config.talker.hiddenSize

        // 1. Tokenize ref_text and target_text. The tokenizer reports
        // `add_prefix_space: false`, so encoding raw text is equivalent to
        // encoding the chat-template-wrapped string and slicing the template
        // tokens off — verified by comparing token IDs against HF's Python
        // tokenizer on the same inputs.
        let refTextTokens = tokenizer.encode(referenceText)
        let targetTextTokens = tokenizer.encode(targetText)

        // 2. TTS special tokens
        let ttsPadEmbed = talker.embedText(
            MLXArray([Int32(CodecTokens.ttsPad)]).expandedDimensions(axis: 0))
        let ttsBosEmbed = talker.embedText(
            MLXArray([Int32(CodecTokens.ttsBos)]).expandedDimensions(axis: 0))
        let ttsEosEmbed = talker.embedText(
            MLXArray([Int32(CodecTokens.ttsEos)]).expandedDimensions(axis: 0))

        // 3. Build text_embed: text_projection(ref_text + target_text) + eos
        let combinedTextIds = (refTextTokens + targetTextTokens).map { Int32($0) }
        let combinedTextArray = MLXArray(combinedTextIds).expandedDimensions(axis: 0)
        let textEmbed = talker.embedText(combinedTextArray)  // [1, T_text, D]
        let textEmbedWithEos = concatenated([textEmbed, ttsEosEmbed], axis: 1)
        let textLens = textEmbedWithEos.dim(1)

        // 4. Build codec_embed: codec_bos + sum_of_all_codebook_embeddings(ref_codes)
        let firstCbCodes = refCodes[0..., 0, 0...]  // [1, T_ref]
        var refCodecEmbed = talker.embedCodec(firstCbCodes)  // [1, T_ref, D]
        for i in 0..<(config.codePredictor.numCodeGroups - 1) {
            let cbCodes = refCodes[0..., i + 1, 0...]  // [1, T_ref]
            refCodecEmbed = refCodecEmbed + codePredictor.codecEmbeddings[i](cbCodes)
        }

        let codecBosEmbed = talker.embedCodec(
            MLXArray([Int32(CodecTokens.codecBos)]).expandedDimensions(axis: 0))
        let codecEmbedICL = concatenated([codecBosEmbed, refCodecEmbed], axis: 1)  // [1, T_ref+1, D]
        let codecLens = codecEmbedICL.dim(1)

        // 5. Non-streaming ICL layout, matching mlx-audio 0.5.4:
        // all reference + target text over codec_pad, followed by reference
        // codec over tts_pad. Keep the complete text context in the prefill
        // instead of element-wise alignment with the reference codec frames.
        let codecPadEmbed = talker.embedCodec(
            MLXArray([Int32(CodecTokens.codecPad)]).expandedDimensions(axis: 0))
        let textWithCodecPad = textEmbedWithEos
            + broadcast(codecPadEmbed, to: [1, textLens, hiddenSize])
        let codecWithTextPad = codecEmbedICL
            + broadcast(ttsPadEmbed, to: [1, codecLens, hiddenSize])
        let iclInputEmbed = concatenated([textWithCodecPad, codecWithTextPad], axis: 1)
        let iclTrailing = ttsPadEmbed

        // 6. Codec prefix. Two layouts (reference parity):
        //   With language id: [codec_think, codec_think_bos, lang_id, codec_think_eos, pad, bos]
        //                     speaker injected after think_eos → 7 tokens.
        //   Without (auto):   [codec_nothink, codec_think_bos, codec_think_eos, pad, bos]
        //                     speaker injected after think_eos → 6 tokens.
        let codecPrefixTokens: [Int32]
        let speakerInjectAt: Int
        if let langId = languageId {
            codecPrefixTokens = [
                Int32(CodecTokens.codecThink),
                Int32(CodecTokens.codecThinkBos),
                Int32(langId),
                Int32(CodecTokens.codecThinkEos),
                Int32(CodecTokens.codecPad),
                Int32(CodecTokens.codecBos),
            ]
            speakerInjectAt = 4
        } else {
            codecPrefixTokens = [
                Int32(CodecTokens.codecNothink),
                Int32(CodecTokens.codecThinkBos),
                Int32(CodecTokens.codecThinkEos),
                Int32(CodecTokens.codecPad),
                Int32(CodecTokens.codecBos),
            ]
            speakerInjectAt = 3
        }
        let codecPrefixArray = MLXArray(codecPrefixTokens).expandedDimensions(axis: 0)
        var codecPrefixEmbed = talker.embedCodec(codecPrefixArray)

        // Inject speaker embedding right after think_eos.
        let spkEmbedReshaped = speakerEmbed.reshaped([1, 1, hiddenSize])
        let part0 = codecPrefixEmbed[0..., 0..<speakerInjectAt, 0...]
        let part1 = codecPrefixEmbed[0..., speakerInjectAt..., 0...]
        codecPrefixEmbed = concatenated([part0, spkEmbedReshaped, part1], axis: 1)

        let codecSuffixEmbed = talker.embedCodec(
            MLXArray([Int32(CodecTokens.codecPad), Int32(CodecTokens.codecBos)]).expandedDimensions(axis: 0))

        // 7. Role embedding: <|im_start|>assistant\n (3 tokens)
        let imStartId: Int32 = 151644
        let assistantId: Int32 = 77091
        let newlineId: Int32 = 198
        let roleTokens = MLXArray([imStartId, assistantId, newlineId]).expandedDimensions(axis: 0)
        let roleEmbed = talker.embedText(roleTokens)  // [1, 3, D]

        // 8. Build pad/bos prefix (text side overlaid with codec prefix)
        let prefixLen = codecPrefixEmbed.dim(1)
        let padCount = prefixLen - 2
        let padEmbeds = broadcast(ttsPadEmbed, to: [1, padCount, hiddenSize])
        let combinedPrefix = concatenated([padEmbeds, ttsBosEmbed], axis: 1)  // [1, prefixLen-1, D]
        let combinedPrefixOverlay = combinedPrefix + codecPrefixEmbed[0..., 0..<(prefixLen - 1), 0...]

        // 9. Full input_embeds: role + codec_prefix + icl_embed
        let inputEmbeds = concatenated(
            [roleEmbed, combinedPrefixOverlay, iclInputEmbed], axis: 1)

        // All text is in the non-streaming prefill; generation receives tts_pad.
        return (inputEmbeds, iclTrailing, ttsPadEmbed)
    }
}
