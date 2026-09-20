import AudioCommon
import XCTest
@testable import AudioServer

final class RealtimeVADTests: XCTestCase {
    func testWebSocketAutomaticallyCommitsAfterVADEndpoint() async throws {
        let port = 19_389
        let vad = ScriptedVAD(
            sampleRate: 24_000,
            chunkSize: 6_000,
            probabilities: [0, 0.8, 0.8, 0.8, 0.1, 0.1])
        let server = AudioServer(
            host: "127.0.0.1",
            port: port,
            realtimeState: FailingRealtimeModelLoading(vadOverride: vad))
        let serverTask = Task { try await server.run() }
        defer { serverTask.cancel() }
        try await Task.sleep(nanoseconds: 500_000_000)

        let socket = URLSession.shared.webSocketTask(
            with: URL(string: "ws://127.0.0.1:\(port)/v1/realtime")!)
        socket.resume()
        defer { socket.cancel(with: .normalClosure, reason: nil) }

        _ = try await receiveJSON(socket)
        try await sendJSON(socket, [
            "type": "session.update",
            "session": [
                "turn_detection": [
                    "type": "server_vad",
                    "prefix_padding_ms": 250,
                    "silence_duration_ms": 250,
                ]
            ]
        ] as [String: Any])
        _ = try await receiveJSON(socket)

        // Deliberately split one PCM16 sample across websocket messages.
        try await sendJSON(socket, [
            "type": "input_audio_buffer.append",
            "audio": Data(repeating: 0, count: 1).base64EncodedString(),
        ])
        try await sendJSON(socket, [
            "type": "input_audio_buffer.append",
            "audio": Data(repeating: 0, count: 36_000 * 2 - 1).base64EncodedString(),
        ])

        let started = try await receiveJSON(socket)
        let stopped = try await receiveJSON(socket)
        let committed = try await receiveJSON(socket)
        let inferenceError = try await receiveJSON(socket)
        XCTAssertEqual(started["type"] as? String, "input_audio_buffer.speech_started")
        XCTAssertEqual(stopped["type"] as? String, "input_audio_buffer.speech_stopped")
        XCTAssertEqual(committed["type"] as? String, "input_audio_buffer.committed")
        XCTAssertEqual(inferenceError["type"] as? String, "error")
        XCTAssertEqual(started["item_id"] as? String, stopped["item_id"] as? String)
        XCTAssertEqual(stopped["item_id"] as? String, committed["item_id"] as? String)
        XCTAssertEqual(started["audio_start_ms"] as? Int, 0)
        XCTAssertEqual(stopped["audio_end_ms"] as? Int, 1_000)
    }

    func testControllerEmitsBoundedPrefixAndClosesAtSpeechEnd() throws {
        let vad = ScriptedVAD(
            sampleRate: 16,
            chunkSize: 4,
            probabilities: [0, 0.8, 0.8, 0.8, 0.1, 0.1])
        let controller = try RealtimeVADController(
            provider: vad,
            config: RealtimeTurnDetectionConfig(
                threshold: 0.5,
                prefixPaddingMilliseconds: 250,
                silenceDurationMilliseconds: 250,
                maxTurnDurationMilliseconds: 5_000),
            inputSampleRate: 16)

        let input = (0..<24).map(Float.init)
        let events = try controller.push(input)

        XCTAssertEqual(events.count, 2)
        guard case .speechStarted(let startMilliseconds) = events[0] else {
            return XCTFail("Expected speech-start event")
        }
        XCTAssertEqual(startMilliseconds, 0)
        guard case .speechEnded(
            let endMilliseconds,
            let audio,
            let forced) = events[1] else {
            return XCTFail("Expected speech-end event")
        }
        XCTAssertEqual(endMilliseconds, 1_000)
        XCTAssertFalse(forced)
        XCTAssertEqual(audio, Array(input.prefix(16)))
    }

    func testSilenceRetentionIsBoundedToDetectionHistory() throws {
        let vad = ScriptedVAD(
            sampleRate: 16,
            chunkSize: 4,
            probabilities: [Float](repeating: 0, count: 100))
        let controller = try RealtimeVADController(
            provider: vad,
            config: RealtimeTurnDetectionConfig(
                threshold: 0.5,
                prefixPaddingMilliseconds: 250,
                silenceDurationMilliseconds: 250,
                maxTurnDurationMilliseconds: 5_000),
            inputSampleRate: 16)

        let events = try controller.push([Float](repeating: 0, count: 400))

        XCTAssertTrue(events.isEmpty)
        XCTAssertLessThanOrEqual(controller.retainedSampleCount, 12)
    }

    func testControllerStatefullyResamplesProtocolAudioForVAD() throws {
        let vad = ScriptedVAD(
            sampleRate: 16_000,
            chunkSize: 512,
            probabilities: [Float](repeating: 0, count: 8))
        let controller = try RealtimeVADController(
            provider: vad,
            config: RealtimeTurnDetectionConfig(),
            inputSampleRate: 24_000)

        _ = try controller.push([Float](repeating: 0.1, count: 768 * 6))
        let processedBeforeCarry = vad.processedSampleCounts.count
        XCTAssertGreaterThanOrEqual(processedBeforeCarry, 5)

        // Converter latency/carry is preserved and completed by later input.
        _ = try controller.push([Float](repeating: 0.1, count: 768))

        XCTAssertGreaterThan(vad.processedSampleCounts.count, processedBeforeCarry)
        XCTAssertTrue(vad.processedSampleCounts.allSatisfy { $0 == 512 })
    }

    func testContinuouslyVoicedTurnsKeepAbsoluteTimeAfterForcedReset() throws {
        let vad = ScriptedVAD(
            sampleRate: 16,
            chunkSize: 4,
            probabilities: [Float](repeating: 0.9, count: 8))
        let controller = try RealtimeVADController(
            provider: vad,
            config: RealtimeTurnDetectionConfig(
                threshold: 0.5,
                prefixPaddingMilliseconds: 0,
                silenceDurationMilliseconds: 250,
                maxTurnDurationMilliseconds: 1_000),
            inputSampleRate: 16)

        let events = try controller.push([Float](repeating: 0.25, count: 32))

        XCTAssertEqual(events.count, 4)
        guard case .speechStarted(let firstStart) = events[0],
              case .speechEnded(let firstEnd, let firstAudio, let firstForced) = events[1],
              case .speechStarted(let secondStart) = events[2],
              case .speechEnded(let secondEnd, let secondAudio, let secondForced) = events[3]
        else { return XCTFail("Expected two duration-limited turns") }
        XCTAssertEqual(firstStart, 0)
        XCTAssertEqual(firstEnd, 1_000)
        XCTAssertEqual(firstAudio.count, 16)
        XCTAssertTrue(firstForced)
        XCTAssertEqual(secondStart, 1_000)
        XCTAssertEqual(secondEnd, 2_000)
        XCTAssertEqual(secondAudio.count, 16)
        XCTAssertTrue(secondForced)
        XCTAssertGreaterThan(vad.resetCount, 0)
        XCTAssertLessThanOrEqual(controller.retainedSampleCount, 12)
    }

    private func receiveJSON(
        _ socket: URLSessionWebSocketTask
    ) async throws -> [String: Any] {
        let message = try await socket.receive()
        guard case .string(let text) = message,
              let data = text.data(using: .utf8),
              let object = try JSONSerialization.jsonObject(with: data)
                as? [String: Any] else {
            XCTFail("Expected a JSON websocket message")
            return [:]
        }
        return object
    }

    private func sendJSON(
        _ socket: URLSessionWebSocketTask,
        _ object: [String: Any]
    ) async throws {
        let data = try JSONSerialization.data(withJSONObject: object)
        try await socket.send(.string(String(decoding: data, as: UTF8.self)))
    }
}

/// Real-model coverage for the exact 24 kHz websocket-audio path used by the
/// server. The isolated E2E runner executes this class in its own process.
final class E2ERealtimeServerVADTests: XCTestCase {
    func testCoreMLServerVADProcessesRealtimeAudio() async throws {
        let provider = try await ModelState().loadVAD()
        let audioURL = try XCTUnwrap(
            Bundle.module.url(forResource: "test_audio", withExtension: "wav"))
        var audio = try AudioFileLoader.load(
            url: audioURL,
            targetSampleRate: 24_000)
        audio.append(contentsOf: [Float](repeating: 0, count: 24_000))
        let audio16k = AudioFileLoader.resample(audio, from: 24_000, to: 16_000)

        for offset in stride(from: 0, to: min(audio16k.count, 512 * 3), by: 512) {
            guard offset + 512 <= audio16k.count else { break }
            _ = provider.processChunk(Array(audio16k[offset..<offset + 512]))
        }
        provider.resetState()

        let rawStarted = CFAbsoluteTimeGetCurrent()
        for offset in stride(from: 0, through: audio16k.count - 512, by: 512) {
            _ = provider.processChunk(Array(audio16k[offset..<offset + 512]))
        }
        let rawElapsed = CFAbsoluteTimeGetCurrent() - rawStarted
        provider.resetState()

        let controller = try RealtimeVADController(
            provider: provider,
            config: RealtimeTurnDetectionConfig(),
            inputSampleRate: 24_000)

        var events: [RealtimeVADEvent] = []
        var appendLatencies: [Double] = []
        let started = CFAbsoluteTimeGetCurrent()
        for offset in stride(from: 0, to: audio.count, by: 480) {
            let end = min(offset + 480, audio.count)
            let appendStarted = CFAbsoluteTimeGetCurrent()
            events.append(contentsOf: try controller.push(Array(audio[offset..<end])))
            appendLatencies.append(CFAbsoluteTimeGetCurrent() - appendStarted)
        }
        let elapsed = CFAbsoluteTimeGetCurrent() - started
        let audioDuration = Double(audio.count) / 24_000
        let sortedLatencies = appendLatencies.sorted()
        let p95Index = min(
            sortedLatencies.count - 1,
            Int(Double(sortedLatencies.count - 1) * 0.95))
        let p95Milliseconds = sortedLatencies[p95Index] * 1_000
        let rtf = elapsed / audioDuration
        let rawRTF = rawElapsed / audioDuration

        let starts = events.compactMap { event -> Int? in
            guard case .speechStarted(let milliseconds) = event else { return nil }
            return milliseconds
        }
        let endings = events.compactMap { event -> (Int, [Float])? in
            guard case .speechEnded(let milliseconds, let audio, _) = event else {
                return nil
            }
            return (milliseconds, audio)
        }

        print(String(
            format: "[server-vad] audio=%.2fs raw-RTF=%.4f server-RTF=%.4f append-p95=%.3fms overhead=%.3fs",
            audioDuration, rawRTF, rtf, p95Milliseconds,
            max(0, elapsed - rawElapsed)))
        XCTAssertFalse(starts.isEmpty, "Real Silero VAD should detect fixture speech")
        XCTAssertFalse(endings.isEmpty, "Trailing silence should close fixture speech")
        XCTAssertTrue(endings.allSatisfy { !$0.1.isEmpty })
        XCTAssertLessThan(rtf, 1, "Server VAD must remain faster than realtime")
    }
}

final class ScriptedVAD: StreamingVADProvider, @unchecked Sendable {
    let inputSampleRate: Int
    let chunkSize: Int
    private var probabilities: [Float]
    private var index = 0
    private(set) var resetCount = 0
    private(set) var processedSampleCounts: [Int] = []

    init(sampleRate: Int, chunkSize: Int, probabilities: [Float]) {
        self.inputSampleRate = sampleRate
        self.chunkSize = chunkSize
        self.probabilities = probabilities
    }

    func processChunk(_ samples: [Float]) -> Float {
        processedSampleCounts.append(samples.count)
        defer { index += 1 }
        return index < probabilities.count ? probabilities[index] : 0
    }

    func resetState() {
        resetCount += 1
    }
}
