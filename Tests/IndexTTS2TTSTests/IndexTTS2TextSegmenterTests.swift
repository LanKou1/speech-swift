@testable import IndexTTS2TTS
import XCTest

final class IndexTTS2TextSegmenterTests: XCTestCase {
    private func tokens(_ pieces: String...) -> [IndexTTS2Token] {
        pieces.enumerated().map { IndexTTS2Token(id: $0.offset, piece: $0.element) }
    }

    private func pieces(_ segments: [[IndexTTS2Token]]) -> [[String]] {
        segments.map { $0.map(\.piece) }
    }

    func testShortTextStaysOneSegment() {
        let input = tokens("▁HELLO", ",", "▁WORLD", ".", "▁HOW", "▁ARE", "▁YOU", "?")
        XCTAssertEqual(pieces(IndexTTS2TextSegmenter.split(input, maxTokens: 120)), [input.map(\.piece)])
    }

    func testSentencesSplitWhenTheyDoNotFitTogether() {
        let input = tokens("▁A", "▁B", "▁C", ".", "▁D", "▁E", "▁F", "?", "▁G", "▁H", "!")
        XCTAssertEqual(
            pieces(IndexTTS2TextSegmenter.split(input, maxTokens: 8)),
            [["▁A", "▁B", "▁C", ".", "▁D", "▁E", "▁F", "?"], ["▁G", "▁H", "!"]])
    }

    func testCommasAndHyphensAreFallbackCuts() {
        let input = tokens("▁A", "▁B", ",", "▁C", "▁D", "-", "▁E", "▁F", ".")
        XCTAssertEqual(
            pieces(IndexTTS2TextSegmenter.split(input, maxTokens: 4)),
            [["▁A", "▁B", ","], ["▁C", "▁D", "-"], ["▁E", "▁F", "."]])
    }

    func testPunctuationFreeRunIsChunkedAtTheLimit() {
        let input = tokens("▁", "你", "▁", "好", "▁", "世", "▁", "界", "▁", "是")
        XCTAssertEqual(
            pieces(IndexTTS2TextSegmenter.split(input, maxTokens: 4)),
            [["▁", "你", "▁", "好"], ["▁", "世", "▁", "界"], ["▁", "是"]])
    }

    func testLengthCutsPreferWordBoundaries() {
        let input = tokens("▁A", "B", "▁C", "D", "▁E", "F")
        XCTAssertEqual(
            pieces(IndexTTS2TextSegmenter.split(input, maxTokens: 4)),
            [["▁A", "B", "▁C", "D"], ["▁E", "F"]])
    }

    func testTrailingApostropheStaysWithItsSentence() {
        let input = tokens("▁HE", "▁SAID", "▁'", "HI", ".", "'", "▁THEN", "▁LEFT", ".")
        XCTAssertEqual(
            pieces(IndexTTS2TextSegmenter.split(input, maxTokens: 6)),
            [["▁HE", "▁SAID", "▁'", "HI", ".", "'"], ["▁THEN", "▁LEFT", "."]])
    }

    func testVeryShortSentenceDoesNotCloseASegment() {
        // Upstream only cuts at sentence punctuation once a run has more than two tokens.
        let input = tokens("▁OK", ".", "▁THEN", "▁WE", "▁GO", ".")
        XCTAssertEqual(
            pieces(IndexTTS2TextSegmenter.split(input, maxTokens: 120)),
            [["▁OK", ".", "▁THEN", "▁WE", "▁GO", "."]])
    }

    func testEmptyInput() {
        XCTAssertEqual(IndexTTS2TextSegmenter.split([], maxTokens: 120).count, 0)
    }
}
