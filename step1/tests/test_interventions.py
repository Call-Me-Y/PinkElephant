from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from intervention_lib import violates, active_rules
from pinkelephant.ftb import TrajStep


class ConstraintTests(unittest.TestCase):
    def test_missing_field_blocks_completion_preserves_correct_values(self):
        rules = {"send_message": {"missing_argument"}}
        self.assertTrue(violates("send_message(to='a@b', subject='x')", rules))
        self.assertFalse(violates("send_message(to='a@b', subject='x', body='y')", rules))
        self.assertFalse(violates("send_message(to='a@b', subject='x', body=", rules))
        self.assertTrue(violates("send_message(to='a@b', subject='x'", rules, ending=True))

    def test_argument_rule_does_not_read_inside_strings(self):
        rules = {"send_message": {"unexpected_argument"}}
        self.assertTrue(violates("send_message(date=", rules))
        self.assertFalse(violates("send_message(body='date=bad', to='x@y', subject='x')", rules))
        self.assertFalse(violates("send_message(sub", rules))

    def test_date_rule_checks_partial_format(self):
        rules = {"list_events": {"format_error"}}
        self.assertTrue(violates("list_events(date='September", rules))
        self.assertTrue(violates("list_events(date='2026-9-14')", rules))
        self.assertFalse(violates("list_events(date='2026-09-14')", rules))
        self.assertFalse(violates("list_events(date='2026-09-", rules))

    def test_constraints_are_tool_specific_and_expire_after_success(self):
        bad = TrajStep("x", "err", False, "missing_argument", "send_message")
        self.assertFalse(violates("list_events()", active_rules([bad])))
        self.assertEqual(active_rules([bad, TrajStep("y", "ok", True)]), {})


if __name__ == "__main__":
    unittest.main()
