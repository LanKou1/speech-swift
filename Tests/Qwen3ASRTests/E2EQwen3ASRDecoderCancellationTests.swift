import XCTest
import MLX
@testable import Qwen3ASR

/// Cooperative cancellation for the Qwen3-ASR MLX decoder loops.
///
/// Uses a weight-free 64-token decoder with no EOS in its vocabulary, so
/// only the token budget or a cancellation checkpoint can stop decoding.
/// No downloads or model weights are needed, but decoder evaluation uses
/// the GPU, so this suite runs separately from the unit job.
final class E2EQwen3ASRDecoderCancellationTests: XCTestCase {
    private static func makeDecoder() -> (QuantizedTextModel, MLXArray) {
        var config = TextDecoderConfig()
        config.vocabSize = 64
        config.hiddenSize = 64
        config.numLayers = 0
        config.intermediateSize = 64
        let decoder = QuantizedTextModel(config: config)
        let logits = MLXArray(Array(repeating: Float(0), count: 64), [1, 1, 64])
        return (decoder, logits)
    }

    /// Runs one of the two decoder loops with an injected checkpoint.
    private static func decode(
        slow: Bool,
        maxTokens: Int,
        checkCancellation: () throws -> Void = {}
    ) throws -> [Int32] {
        let (decoder, logits) = makeDecoder()
        if slow {
            return try Qwen3ASRModel.generateSlow(
                textDecoder: decoder, initialLogits: logits, cache: [],
                maxTokens: maxTokens,
                options: Qwen3DecodingOptions(noRepeatNgramSize: 3),
                checkCancellation: checkCancellation)
        }
        return try Qwen3ASRModel.generateGreedyAsyncEval(
            textDecoder: decoder, initialLogits: logits, cache: [],
            maxTokens: maxTokens, checkCancellation: checkCancellation)
    }

    // MARK: - Checkpoint accounting (deterministic, no timing)

    func testThrowingCheckpointStopsBothLoopsBeforeTheNextToken() {
        for slow in [false, true] {
            var checkpoints = 0
            XCTAssertThrowsError(
                try Self.decode(slow: slow, maxTokens: 100) {
                    checkpoints += 1
                    if checkpoints == 4 { throw CancellationError() }
                },
                "slow=\(slow)"
            ) { error in
                XCTAssertTrue(error is CancellationError, "slow=\(slow): \(error)")
            }
            XCTAssertEqual(checkpoints, 4, "slow=\(slow): no checkpoint may run after the throw")
        }
    }

    func testBudgetTerminationRunsExactlyOneCheckpointPerToken() throws {
        for slow in [false, true] {
            var checkpoints = 0
            let tokens = try Self.decode(slow: slow, maxTokens: 5) { checkpoints += 1 }
            XCTAssertEqual(tokens.count, 5, "slow=\(slow)")
            XCTAssertEqual(
                checkpoints, 5,
                "slow=\(slow): one checkpoint per submitted token, none after the last")
        }
    }

    // MARK: - Contract: the synchronous loops ignore task cancellation

    func testDefaultCheckpointIgnoresCancelledTask() async throws {
        for slow in [false, true] {
            let tokens = try await Task.detached { () throws -> [Int32] in
                withUnsafeCurrentTask { $0?.cancel() }
                return try Self.decode(slow: slow, maxTokens: 8)
            }.value
            XCTAssertEqual(
                tokens.count, 8,
                "slow=\(slow): synchronous decode must run to its budget inside a cancelled task")
        }
    }

    // MARK: - Contract: Task.checkCancellation surfaces as CancellationError

    func testPrecancelledTaskThrowsBeforeDecoding() async {
        for slow in [false, true] {
            let result = await Task.detached { () throws -> [Int32] in
                withUnsafeCurrentTask { $0?.cancel() }
                return try Self.decode(slow: slow, maxTokens: 8) { try Task.checkCancellation() }
            }.result
            switch result {
            case .success(let tokens):
                XCTFail("slow=\(slow): expected CancellationError, decoded \(tokens.count) tokens")
            case .failure(let error):
                XCTAssertTrue(error is CancellationError, "slow=\(slow): \(error)")
            }
        }
    }

    func testCancellingTheTaskStopsInFlightDecoding() async throws {
        for slow in [false, true] {
            // Every one of the 10,000 steps synchronises with the GPU, so the
            // loop cannot drain its budget before the cancel below lands; a
            // success here means the checkpoint was never consulted.
            let task = Task.detached { () throws -> [Int32] in
                try Self.decode(slow: slow, maxTokens: 10_000) { try Task.checkCancellation() }
            }
            try await Task.sleep(for: .milliseconds(100))
            task.cancel()
            switch await task.result {
            case .success(let tokens):
                XCTFail("slow=\(slow): expected CancellationError, decoded \(tokens.count) tokens")
            case .failure(let error):
                XCTAssertTrue(error is CancellationError, "slow=\(slow): \(error)")
            }
        }
    }
}
