import XCTest
import CoreML
@testable import SpeechRestoration

/// Unit tests for Sidon config + variant wiring (no model download).
final class SpeechRestorationConfigTests: XCTestCase {

    func testVariantSubfoldersAndRepo() {
        XCTAssertEqual(SidonVariant.fp16.subfolder, "fp16")
        XCTAssertEqual(SidonVariant.int8.subfolder, "int8")
        XCTAssertEqual(SidonVariant.fp16.defaultModelId, "aufklarer/Sidon-CoreML")
        XCTAssertEqual(SidonVariant.allCases.count, 2)
        XCTAssertEqual(SidonVariant(rawValue: "fp16"), .fp16)
        XCTAssertEqual(SidonVariant(rawValue: "int8"), .int8)
        XCTAssertNil(SidonVariant(rawValue: "int4"))
    }

    func testDefaultConfigMatchesExport() {
        let c = SidonConfig.default
        XCTAssertEqual(c.inputSampleRate, 16_000)
        XCTAssertEqual(c.outputSampleRate, 48_000)
        XCTAssertEqual(c.frames, 499)
        XCTAssertEqual(c.hiddenSize, 1024)
        XCTAssertEqual(c.featureDim, 160)
        // The fixed window is exactly 10 s of 16 kHz audio → 499 stacked frames.
        XCTAssertEqual(c.windowSamples, 160_000)
        XCTAssertEqual(c.outputSamplesPerWindow, 479_014)
    }

    func testWindowSpansTenSeconds() {
        let c = SidonConfig.default
        // 499 stacked frames ⇒ 998 mel frames; 1 + (N-400)/160 = 998 ⇒ N≈159_920;
        // the extractor yields exactly 499 stacked frames at 160_000 samples.
        let (_, frames) = SeamlessM4TFrontEnd.inputFeatures(
            audio: [Float](repeating: 0, count: c.windowSamples))
        XCTAssertEqual(frames, c.frames,
            "A full \(c.windowSamples)-sample window must produce exactly \(c.frames) frames")
    }

    func testEngineConstants() {
        XCTAssertEqual(SpeechRestorer.inputSampleRate, 16_000)
        XCTAssertEqual(SpeechRestorer.outputSampleRate, 48_000)
        XCTAssertEqual(SpeechRestorer.defaultModelId, "aufklarer/Sidon-CoreML")
    }

    // MARK: - Compute placement

    /// The vocoder must never default to a Neural-Engine-eligible unit: its ANE
    /// compile takes minutes and can fail (issue #464). The predictor may.
    func testDefaultPlacementKeepsVocoderOffTheNeuralEngine() {
        let p = SidonComputePlacement.default
        XCTAssertEqual(p.predictor, .all)
        XCTAssertEqual(p.vocoder, .cpuAndGPU)
        XCTAssertEqual(SidonComputePlacement.defaultVocoder, .cpuAndGPU)
        XCTAssertEqual(
            SidonComputePlacement.resolve(computeUnits: nil, predictor: nil, vocoder: nil), p,
            "no parameters must resolve to the shipped defaults")
    }

    func testUniformComputeUnitsApplyToBothStages() {
        let p = SidonComputePlacement.resolve(computeUnits: .cpuOnly, predictor: nil, vocoder: nil)
        XCTAssertEqual(p, .uniform(.cpuOnly))
        XCTAssertEqual(p.predictor, .cpuOnly)
        XCTAssertEqual(p.vocoder, .cpuOnly)
    }

    func testPerStageOverridesWinOverUniform() {
        let p = SidonComputePlacement.resolve(
            computeUnits: .cpuOnly, predictor: .cpuAndNeuralEngine, vocoder: nil)
        XCTAssertEqual(p.predictor, .cpuAndNeuralEngine)
        XCTAssertEqual(p.vocoder, .cpuOnly)

        let q = SidonComputePlacement.resolve(computeUnits: nil, predictor: nil, vocoder: .all)
        XCTAssertEqual(q.predictor, .all, "unset stage keeps its default")
        XCTAssertEqual(q.vocoder, .all, "explicit opt-in to ANE for the vocoder is allowed")
    }

    func testRestorerRecordsPlacement() {
        let r = SpeechRestorer(
            predictor: nil, vocoder: nil, config: .default, variant: .fp16,
            placement: .uniform(.cpuAndGPU), loaded: false)
        XCTAssertEqual(r.placement, .uniform(.cpuAndGPU))
        let d = SpeechRestorer(predictor: nil, vocoder: nil, config: .default, variant: .fp16)
        XCTAssertEqual(d.placement, .default)
    }
}
