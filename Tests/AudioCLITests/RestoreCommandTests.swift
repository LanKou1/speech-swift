import XCTest
import ArgumentParser
@testable import AudioCLILib

final class RestoreCommandTests: XCTestCase {

    func testDefaultsLeavePlacementToTheEngine() throws {
        let cmd = try AudioCLI.parseAsRoot(["restore", "noisy.wav"])
        let restore = try XCTUnwrap(cmd as? RestoreCommand)
        XCTAssertEqual(restore.audioFile, "noisy.wav")
        XCTAssertEqual(restore.variant, "fp16")
        XCTAssertNil(restore.output)
        XCTAssertNil(restore.computeUnits, "no flag → per-stage engine defaults")
    }

    func testParsesComputeUnitsAlongsideOtherFlags() throws {
        let cmd = try AudioCLI.parseAsRoot([
            "restore", "noisy.wav", "--compute-units", "gpu",
            "--variant", "int8", "-o", "clean.wav",
        ])
        let restore = try XCTUnwrap(cmd as? RestoreCommand)
        XCTAssertEqual(restore.computeUnits, "gpu")
        XCTAssertEqual(restore.variant, "int8")
        XCTAssertEqual(restore.output, "clean.wav")
    }

    func testAcceptsEveryComputeUnitsName() throws {
        for name in ["ane", "gpu", "cpu", "all", "cpuAndGPU", "CPU"] {
            let cmd = try AudioCLI.parseAsRoot(["restore", "x.wav", "--compute-units", name])
            XCTAssertEqual((cmd as? RestoreCommand)?.computeUnits, name)
        }
    }

    func testRejectsUnknownComputeUnitsAtParseTime() {
        XCTAssertThrowsError(
            try AudioCLI.parseAsRoot(["restore", "x.wav", "--compute-units", "metal"])
        ) { error in
            XCTAssertEqual(AudioCLI.exitCode(for: error), .validationFailure)
            XCTAssertTrue(AudioCLI.message(for: error).contains("compute units"))
        }
    }

    func testRejectsUnknownVariantAtParseTime() {
        XCTAssertThrowsError(
            try AudioCLI.parseAsRoot(["restore", "x.wav", "--variant", "int4"])
        ) { error in
            XCTAssertEqual(AudioCLI.exitCode(for: error), .validationFailure)
        }
    }

    func testHelpDocumentsComputeUnits() {
        let help = RestoreCommand.helpMessage()
        XCTAssertTrue(help.contains("--compute-units"))
        XCTAssertTrue(help.contains("vocoder"))
    }
}
