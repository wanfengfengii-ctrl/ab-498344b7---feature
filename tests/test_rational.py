import unittest
from fractions import Fraction

from app.rational import RationalFormatError, format_rational, parse_rational


class ParseRationalValidTests(unittest.TestCase):
    def test_json_integer(self):
        self.assertEqual(parse_rational(3), Fraction(3))
        self.assertEqual(parse_rational(-7), Fraction(-7))
        self.assertEqual(parse_rational(0), Fraction(0))

    def test_integer_string(self):
        self.assertEqual(parse_rational("3"), Fraction(3))
        self.assertEqual(parse_rational("-7"), Fraction(-7))
        self.assertEqual(parse_rational("+5"), Fraction(5))

    def test_reduced_fraction_string(self):
        self.assertEqual(parse_rational("3/4"), Fraction(3, 4))
        self.assertEqual(parse_rational("-3/4"), Fraction(-3, 4))
        self.assertEqual(parse_rational("0/1"), Fraction(0))
        self.assertEqual(parse_rational("5/1"), Fraction(5))
        self.assertEqual(parse_rational("22/7"), Fraction(22, 7))

    def test_huge_integers(self):
        big = 10 ** 60 + 1
        self.assertEqual(parse_rational(big), Fraction(big))
        self.assertEqual(parse_rational(str(big)), Fraction(big))


class ParseRationalInvalidTests(unittest.TestCase):
    def assert_rejected(self, node):
        with self.assertRaises(RationalFormatError):
            parse_rational(node)

    def test_floats_rejected(self):
        self.assert_rejected(0.5)
        self.assert_rejected(2.0)   # integral floats are still floats
        self.assert_rejected(-1e3)

    def test_bool_rejected(self):
        self.assert_rejected(True)
        self.assert_rejected(False)

    def test_unreduced_fraction_rejected(self):
        self.assert_rejected("2/4")
        self.assert_rejected("0/5")
        self.assert_rejected("-9/6")

    def test_non_positive_denominator_rejected(self):
        self.assert_rejected("1/0")
        self.assert_rejected("1/-2")
        self.assert_rejected("-1/-2")

    def test_malformed_strings_rejected(self):
        for bad in ("", " ", "1.5", "1e3", "abc", "1/2/3", "1 /2", "1/ 2",
                    " 1", "1 ", "0x10", "1,5", "/", "+", "-", "1/"):
            self.assert_rejected(bad)

    def test_other_types_rejected(self):
        self.assert_rejected(None)
        self.assert_rejected([1])
        self.assert_rejected({"p": 1})


class FormatRationalTests(unittest.TestCase):
    def test_canonical_form(self):
        self.assertEqual(format_rational(Fraction(3)), "3")
        self.assertEqual(format_rational(Fraction(-7)), "-7")
        self.assertEqual(format_rational(Fraction(6, 8)), "3/4")
        self.assertEqual(format_rational(Fraction(-2, 4)), "-1/2")
        self.assertEqual(format_rational(Fraction(0)), "0")


if __name__ == "__main__":
    unittest.main()
