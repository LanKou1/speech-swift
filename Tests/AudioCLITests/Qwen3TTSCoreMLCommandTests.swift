import XCTest
import ArgumentParser
@testable import AudioCLILib

final class Qwen3TTSCoreMLCommandTests: XCTestCase {
    func testKeepsExistingModelAsDefault() throws {
        let command = try XCTUnwrap(AudioCLI.parseAsRoot(["qwen3-tts-coreml", "Hello"]) as? Qwen3TTSCoreMLCommand)
        XCTAssertEqual(command.model, "aufklarer/Qwen3-TTS-CoreML")
        XCTAssertNil(command.modelDirectory)
        XCTAssertNil(command.speakerEmbedding)
        XCTAssertEqual(command.maxTokens, 125)
    }

    func testParsesLargeBundleAndPreparedSpeaker() throws {
        let command = try XCTUnwrap(AudioCLI.parseAsRoot([
            "qwen3-tts-coreml", "Hello", "--model", "aufklarer/Qwen3-TTS-1.7B-CoreML",
            "--model-directory", "/tmp/bundle", "--speaker-embedding", "/tmp/speaker.npy",
            "--max-tokens", "50", "--temperature", "0",
        ]) as? Qwen3TTSCoreMLCommand)
        XCTAssertEqual(command.model, "aufklarer/Qwen3-TTS-1.7B-CoreML")
        XCTAssertEqual(command.modelDirectory, "/tmp/bundle")
        XCTAssertEqual(command.speakerEmbedding, "/tmp/speaker.npy")
        XCTAssertEqual(command.maxTokens, 50)
        XCTAssertEqual(command.temperature, 0)
    }
}
