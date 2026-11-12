from pathlib import Path
import json
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pe_feedback.bridge import Task, make_env, trajectory_step
from pe_feedback.common import PE, read_rows
from pe_feedback.evidence import Evidence
from pe_feedback.feedback import FeedbackEngine, compose
from pe_feedback.experiment import collect, replay
from slmecho.harness import TrajStep

GOAL = "Send Alice Example the contents of notes/agenda.md. Use 'agenda' as the subject."
CONTACT = TrajStep("find_contact(name='Alice Example')", "{'name': 'Alice Example', 'email': 'alice@example.com', 'team': 'design'}", True, tool="find_contact")
READ = TrajStep("read_file(path='notes/agenda.md')", "Verified original document", True, tool="read_file")
BAD = TrajStep("send_message(to='WRONG_SECRET', subject='agenda')", "error echoes WRONG_SECRET", False, "missing_argument", "send_message")


class MonitorTests(unittest.TestCase):
    def test_six_public_goal_parsers_and_completion_witnesses(self):
        kinds = set()
        for case in read_rows(PE / "step1" / "data" / "pilot.jsonl"):
            env, task = make_env(), Task(**case["task"])
            env.reset(task)
            steps = [trajectory_step(action, env.step(task, action)) for action in task.gold_actions]
            state = Evidence(task.goal, steps)
            self.assertTrue(state.completed, case["case_id"])
            kinds.add(state.goal.kind)
        self.assertEqual(len(kinds), 6)

    def test_failed_values_do_not_become_facts(self):
        state = Evidence(GOAL, [BAD])
        self.assertIsNone(state.get("contact_email"))
        card, _ = FeedbackEngine().issue(GOAL, [CONTACT, READ, BAD], "grounded")
        self.assertIn("alice@example.com", card["text"])
        self.assertNotIn("WRONG_SECRET", json.dumps(card))
        self.assertEqual({f["source_kind"] for f in card["verified_facts"]}, {"user_goal", "tool_success"})

    def test_error_surface_invariance(self):
        other = TrajStep("send_message(subject='agenda', to='ANOTHER_BAD_VALUE')", "another raw error", False, "missing_argument", "send_message")
        a, _ = FeedbackEngine().issue(GOAL, [CONTACT, READ, BAD], "grounded")
        b, _ = FeedbackEngine().issue(GOAL, [CONTACT, READ, other], "grounded")
        self.assertEqual(a, b)

    def test_known_evidence_changes_next_subgoal(self):
        a, _ = FeedbackEngine().issue(GOAL, [BAD], "grounded")
        b, _ = FeedbackEngine().issue(GOAL, [CONTACT, BAD], "grounded")
        c, _ = FeedbackEngine().issue(GOAL, [CONTACT, READ, BAD], "grounded")
        self.assertEqual([a["next_tool"], b["next_tool"], c["next_tool"]], ["find_contact", "read_file", "send_message"])

    def test_unrelated_success_does_not_close_goal(self):
        other = TrajStep("list_files(folder='reports')", "['reports/x.md']", True, tool="list_files")
        state = Evidence(GOAL, [CONTACT, READ, BAD, other])
        self.assertFalse(state.completed)
        self.assertEqual(state.next_step()[0], "send_message")

    def test_wrong_subject_is_not_completion(self):
        wrong = TrajStep("send_message(to='alice@example.com', subject='wrong', body='Verified original document')", "sent", True, tool="send_message")
        self.assertFalse(Evidence(GOAL, [CONTACT, READ, wrong]).completed)
        good = TrajStep("send_message(to='alice@example.com', subject='agenda', body='Verified original document')", "sent", True, tool="send_message")
        self.assertTrue(Evidence(GOAL, [CONTACT, READ, good]).completed)
        state = Evidence(GOAL, [CONTACT, READ, good, TrajStep("list_files(folder='notes')", "[]", True)])
        self.assertEqual(state.facts["completion_receipt"].source_id, "obs:3")

    def test_source_mutation_invalidates_old_read(self):
        write = TrajStep("write_file(path='notes/agenda.md', content='new')", "wrote 3 characters", True, tool="write_file")
        state = Evidence(GOAL, [CONTACT, READ, write, BAD])
        self.assertIsNone(state.get("body"))
        card, _ = FeedbackEngine().issue(GOAL, [CONTACT, READ, write, BAD], "grounded")
        self.assertNotIn(READ.observation, card["text"])
        self.assertEqual(card["next_tool"], "read_file")

    def test_history_detects_ignored_hint(self):
        engine = FeedbackEngine()
        engine.issue(GOAL, [BAD], "adaptive")
        other = TrajStep("list_files(folder='notes')", "[]", True, tool="list_files")
        card, _ = engine.issue(GOAL, [BAD, other, BAD], "adaptive")
        self.assertEqual(card["previous_hint_outcome"], "requested_step_not_observed")

    def test_history_detects_followed_but_irrelevant_lookup(self):
        engine = FeedbackEngine()
        engine.issue(GOAL, [BAD], "adaptive")
        other = TrajStep("find_contact(name='Other Person')", "{'name':'Other Person','email':'other@example.com'}", True, tool="find_contact")
        card, _ = engine.issue(GOAL, [BAD, other, BAD], "adaptive")
        self.assertEqual(card["previous_hint_outcome"], "requested_step_without_observable_progress")

    def test_history_retains_progress_after_success(self):
        engine = FeedbackEngine()
        engine.issue(GOAL, [BAD], "adaptive")
        card, _ = engine.issue(GOAL, [BAD, CONTACT, BAD], "adaptive")
        self.assertEqual(card["previous_hint_outcome"], "observable_progress")
        self.assertEqual(card["next_tool"], "read_file")

    def test_budget_uses_pointer_without_partial_value(self):
        long_read = TrajStep(READ.action, "unique verified content " * 100, True, tool="read_file")
        card, _ = FeedbackEngine().issue(GOAL, [CONTACT, long_read, BAD], "grounded", len, 1100)
        self.assertLessEqual(len(card["text"]), 1100)
        self.assertIn("body", card["budget_omissions"])
        self.assertIn("obs:2", card["text"])
        self.assertNotIn("unique verified content", card["text"])

    def test_unsupported_goal_falls_back_without_invention(self):
        msgs, card, state = compose("public tools", "An unsupported task", [BAD], "grounded", FeedbackEngine())
        self.assertIsNone(card)
        self.assertFalse(state["supported"])
        self.assertNotIn(BAD.action, str(msgs))

    def test_shared_failure_only_trigger(self):
        card, _ = FeedbackEngine().issue(GOAL, [CONTACT], "adaptive")
        self.assertIsNone(card)

    def test_facts_only_contains_no_repair_subgoal(self):
        card, _ = FeedbackEngine().issue(GOAL, [CONTACT, READ, BAD], "facts_only")
        self.assertIsNone(card["next_tool"])
        self.assertNotIn("Next subgoal", card["text"])

    def test_natural_collector_excludes_injected_continuations(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "source.jsonl"
            source.write_text(json.dumps({"mode": "controlled", "condition": "abstract"}) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                collect(source, str(Path(folder) / "result.jsonl"))

    def test_natural_checkpoint_can_be_replayed(self):
        case = read_rows(PE / "step1" / "data" / "pilot.jsonl")[0]
        env, task = make_env(), Task(**case["task"])
        env.reset(task)
        step = trajectory_step("nonexistent_tool()", env.step(task, "nonexistent_tool()"))
        row = {"mode": "free", "condition": "abstract", "cluster_id": case["cluster_id"], "task": case["task"], "turns": [{"executed_step": step.__dict__}]}
        with tempfile.TemporaryDirectory() as folder:
            source, target = Path(folder) / "source.jsonl", Path(folder) / "natural.jsonl"
            source.write_text(json.dumps(row) + "\n", encoding="utf-8")
            collect(source, str(target))
            _, _, restored = replay(read_rows(target)[0])
            self.assertEqual(restored[-1].observation, step.observation)


if __name__ == "__main__":
    unittest.main()
