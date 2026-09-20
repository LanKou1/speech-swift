@testable import IndexTTS2TTS
import XCTest

final class IndexTTS2TextNormalizerTests: XCTestCase {
    private func normalize(_ text: String) -> String {
        IndexTTS2TextNormalizer.normalize(text)
    }

    func testChineseCardinals() {
        XCTAssertEqual(normalize("0个"), "零个")
        XCTAssertEqual(normalize("10个"), "十个")
        XCTAssertEqual(normalize("15个"), "十五个")
        XCTAssertEqual(normalize("110个"), "一百一十个")
        XCTAssertEqual(normalize("200个"), "二百个")
        XCTAssertEqual(normalize("1001个"), "一千零一个")
        XCTAssertEqual(normalize("10001个"), "一万零一个")
        XCTAssertEqual(normalize("100000个"), "十万个")
        XCTAssertEqual(normalize("1234567个"), "一百二十三万四千五百六十七个")
        XCTAssertEqual(normalize("100000001个"), "一亿零一个")
        XCTAssertEqual(normalize("一共1,000个"), "一共一千个")
        XCTAssertEqual(normalize("编号007"), "编号零零七")
    }

    func testChineseDatesTimesAndUnits() {
        XCTAssertEqual(normalize("今天是2024年3月5日"), "今天是二零二四年三月五日")
        XCTAssertEqual(normalize("98年出生"), "九八年出生")
        XCTAssertEqual(normalize("２０２４年"), "二零二四年")
        XCTAssertEqual(normalize("10:30开会，10:05结束，11:00下班"), "十点三十分开会，十点零五分结束，十一点下班")
        XCTAssertEqual(normalize("第1名和第2名"), "第一名和第二名")
        XCTAssertEqual(normalize("50%折扣"), "百分之五十折扣")
        XCTAssertEqual(normalize("圆周率是3.14"), "圆周率是三点一四")
        XCTAssertEqual(normalize("1/2杯"), "二分之一杯")
        XCTAssertEqual(normalize("$100和¥200"), "一百美元和二百元")
        XCTAssertEqual(normalize("温度-5度"), "温度负五度")
        XCTAssertEqual(normalize("电话13800138000"), "电话一三八零零一三八零零零")
    }

    func testEnglishCardinalsAndYears() {
        XCTAssertEqual(normalize("I have 3 cats"), "I have three cats")
        XCTAssertEqual(normalize("123456 people"), "one hundred twenty three thousand four hundred fifty six people")
        XCTAssertEqual(normalize("1,000,000 views"), "one million views")
        XCTAssertEqual(normalize("in 1995"), "in nineteen ninety five")
        XCTAssertEqual(normalize("in 1900"), "in nineteen hundred")
        XCTAssertEqual(normalize("in 1905"), "in nineteen oh five")
        XCTAssertEqual(normalize("in 2000"), "in two thousand")
        XCTAssertEqual(normalize("in 2005"), "in two thousand five")
        XCTAssertEqual(normalize("in 2024"), "in twenty twenty four")
        XCTAssertEqual(normalize("3000 units"), "three thousand units")
        XCTAssertEqual(normalize("agent 007"), "agent zero zero seven")
        XCTAssertEqual(normalize("A2024B"), "A two thousand twenty four B")
    }

    func testEnglishOrdinalsMoneyTimeAndPercent() {
        XCTAssertEqual(normalize("the 21st century"), "the twenty first century")
        XCTAssertEqual(
            normalize("2nd, 3rd, 4th, 12th, 20th, 100th"),
            "second, third, fourth, twelfth, twentieth, one hundredth")
        XCTAssertEqual(normalize("50% off"), "fifty percent off")
        XCTAssertEqual(
            normalize("$3.50 and $1 and $0.99"),
            "three dollars fifty cents and one dollar and ninety nine cents")
        XCTAssertEqual(normalize("at 10:30, 10:05, 10:00"), "at ten thirty, ten oh five, ten o'clock")
        XCTAssertEqual(normalize("pi is 3.14"), "pi is three point one four")
        XCTAssertEqual(normalize("it was -5 degrees"), "it was minus five degrees")
    }

    func testPathSelectionAndPassThrough() {
        // Han text, or text without Latin letters, takes the Chinese readings.
        XCTAssertEqual(normalize("Hello 你好 2"), "Hello 你好 二")
        XCTAssertEqual(normalize("2"), "二")
        XCTAssertEqual(normalize("no digits here"), "no digits here")
        XCTAssertEqual(normalize("你好"), "你好")
        XCTAssertEqual(normalize(""), "")
    }
}
