import XCTest
@testable import SupertonicTTS

/// Offline unit tests for the G2P-free front-end (no model download).
final class SupertonicTokenizerTests: XCTestCase {
    /// Identity table over the BMP: codepoint == id.
    private func identityTokenizer() -> SupertonicTokenizer {
        SupertonicTokenizer(indexer: (0..<65536).map { Int32($0) })
    }

    func testAvailableLanguages() {
        let t = identityTokenizer()
        XCTAssertTrue(t.supports("en"))
        XCTAssertTrue(t.supports("de"))
        XCTAssertTrue(t.supports("ru"))
        XCTAssertTrue(t.supports("ko"))
        XCTAssertFalse(t.supports("zz"))
        XCTAssertFalse(t.supports("zh"))  // Supertonic excludes zh
    }

    func testUnsupportedLanguageThrows() {
        let t = identityTokenizer()
        XCTAssertThrowsError(try t.process("hi", lang: "zz", textLength: 128))
    }

    func testLangWrapAndTerminalPunctuation() throws {
        let t = identityTokenizer()
        let tok = try t.process("hi", lang: "en", textLength: 128)
        // wrapped = "<en>hi.</en>" → ids start with '<','e','n','>'
        XCTAssertEqual(Array(tok.ids.prefix(4)),
                       [Int32(UInt32("<")), Int32(UInt32("e")), Int32(UInt32("n")), Int32(UInt32(">"))])
        XCTAssertEqual(tok.mask[0], 1.0)
    }

    /// NFKD keystone: "Käse" must decompose ä → 'a' + combining diaeresis (U+0308).
    func testNFKDDecomposesUmlaut() throws {
        let t = identityTokenizer()
        let tok = try t.process("Käse", lang: "de", textLength: 128)
        // wrapped = "<de>" + "Ka◌̈se." + "</de>" → index 4 = 'K'? "<de>" is 4 codepoints (0..3),
        // so position 4 is 'K', 5 'a', 6 = U+0308 (combining diaeresis).
        XCTAssertEqual(tok.ids[4], Int32(UInt32("K")))
        XCTAssertEqual(tok.ids[5], Int32(UInt32("a")))
        XCTAssertEqual(tok.ids[6], 0x0308, "NFKD must split ä into a + combining diaeresis")
    }

    func testUnknownCodepointIsMinusOne() throws {
        // indexer covers only 0..127; everything above is unknown (-1).
        let t = SupertonicTokenizer(indexer: (0..<128).map { Int32($0) })
        let tok = try t.process("é", lang: "fr", textLength: 64)
        // After NFKD "é" → 'e' + U+0301; 'e' is in-table, U+0301 (769) is out → -1.
        XCTAssertTrue(tok.ids.contains(Int32(UInt32("e"))))
        XCTAssertTrue(tok.ids.contains(-1), "Out-of-table codepoints must resolve to -1, never crash")
    }

    func testChunkingStaysWithinTextLength() {
        let t = identityTokenizer()
        let long = String(repeating: "word ", count: 200)
        let chunks = t.chunk(long, lang: "en", textLength: 128)
        XCTAssertGreaterThan(chunks.count, 1, "Long text must split into multiple chunks")
        for c in chunks {
            // wrapped length must fit the fixed text graph (T + a little slack for the tag).
            XCTAssertLessThanOrEqual(c.unicodeScalars.count + 2 * 2 + 5, 128 + 1)
        }
    }

    func testNFKDGrowthRespectsTextCapacity() throws {
        let t = identityTokenizer()
        // 19 × "élève": 114 raw scalars — under the 118 raw-scalar budget — but NFKD splits every
        // accent off (é → e + U+0301), so the wrapped form is 161 tokens and process() used to
        // drop the end of the sentence silently.
        let s = (0..<19).map { _ in "élève" }.joined(separator: " ") + "."
        XCTAssertLessThanOrEqual(s.unicodeScalars.count, 118)
        XCTAssertGreaterThan(try t.wrappedLength(s, lang: "fr"), 128)

        let chunks = t.chunk(s, lang: "fr", textLength: 128)
        XCTAssertGreaterThanOrEqual(chunks.count, 2)
        for c in chunks {
            let wrapped = try t.wrappedLength(c, lang: "fr")
            XCTAssertLessThanOrEqual(wrapped, 128)
            let real = try t.process(c, lang: "fr", textLength: 128).mask.filter { $0 > 0 }.count
            XCTAssertEqual(real, wrapped, "nothing may be truncated: \(c)")
        }
        XCTAssertEqual(chunks.joined(separator: " "), s, "nothing lost or reordered")
    }

    func testOversizeSentenceSplitsBalancedAtClauses() throws {
        let t = identityTokenizer()
        // Six 43-scalar clauses = one 269-scalar sentence, far past the 128-token text length.
        // It must be cut — but at commas, in balanced pieces, never leaving a stub.
        let clause = "the quick brown fox jumps over the lazy dog"
        let s = (0..<6).map { _ in clause }.joined(separator: ", ") + "."
        let chunks = t.chunk(s, lang: "en", textLength: 128)
        XCTAssertGreaterThanOrEqual(chunks.count, 3)
        for c in chunks {
            XCTAssertLessThanOrEqual(try t.wrappedLength(c, lang: "en"), 128)
            XCTAssertGreaterThanOrEqual(c.unicodeScalars.count, 40, "no stub pieces: \(c)")
            XCTAssertTrue(c.hasSuffix(",") || c.hasSuffix("."), "cut at a clause boundary: \(c)")
        }
        XCTAssertEqual(chunks.joined(separator: " "), s)
    }

    func testSentencesPackUpToBudget() {
        let t = identityTokenizer()
        // helper.py parity: sentences share a chunk while len(cur) + 1 + len(s) <= budget (118).
        let a = String(repeating: "a", count: 50) + "."
        let b = String(repeating: "b", count: 50) + "."
        XCTAssertEqual(t.chunk(a + " " + b, lang: "fr", textLength: 128), [a + " " + b])  // 103
        let c = String(repeating: "c", count: 70) + "."
        XCTAssertEqual(t.chunk(a + " " + c, lang: "fr", textLength: 128), [a, c])         // 123 > 118
    }

    func testBisectPrefersSentenceThenClauseThenWord() {
        typealias T = SupertonicTokenizer
        let two = T.bisect("Cet été, j'ai passé mes vacances à la campagne. J'ai loué une maison.", minScalars: 8)
        XCTAssertEqual(two?.0, "Cet été, j'ai passé mes vacances à la campagne.")
        XCTAssertEqual(two?.1, "J'ai loué une maison.")
        let clause = T.bisect("Chaque matin, je me levais tôt pour faire une longue promenade.", minScalars: 8)
        XCTAssertEqual(clause?.0, "Chaque matin,")
        let word = T.bisect("J'ai aussi passé du temps à lire des livres sous les arbres.", minScalars: 8)
        XCTAssertNotNil(word)
        XCTAssertTrue(word!.0.unicodeScalars.count >= 20 && word!.1.unicodeScalars.count >= 20, "balanced: \(word!)")
        XCTAssertNil(T.bisect("Bien sûr !", minScalars: 8), "too short to cut")
    }
}
