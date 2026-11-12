"""Protocol checks: no paid calls and no downloaded model required."""
from dataclasses import asdict
from pathlib import Path
import io
import json
import sys
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pinkelephant.analysis import clustered_interval
from pinkelephant.backends import ChatAPIBackend
from pinkelephant.common import PROJECT, read_jsonl
from pinkelephant.experiment import metrics
from pinkelephant.feedback import correction_card, literal_leak_audit, render
from pinkelephant.ftb import Task, TrajStep, canonical, make_env, trajectory_step


class ProtocolTests(unittest.TestCase):
    def test_equivalent_calls_are_repeats(self):
        self.assertEqual(canonical("send_message(to='a@b.com', subject='x', body='y')"),
                         canonical('send_message(body="y",to="a@b.com",subject="x")'))
        self.assertNotEqual(canonical("read_file(path='a')"), canonical("read_file(path='b')"))

    def test_card_uses_schema_without_copying_failed_values(self):
        step = TrajStep("send_message(to='SENSITIVE_RECIPIENT', subject='SENSITIVE_SUBJECT')",
                        "Traceback echoes SENSITIVE_RECIPIENT", False, "missing_argument", "send_message")
        card = correction_card([step], True)
        self.assertIn("body", card["positive_guidance"])
        self.assertNotIn("SENSITIVE", str(card))
        self.assertFalse(literal_leak_audit([step], card)["full_failed_call_present"])
        self.assertEqual(literal_leak_audit([step], card)["argument_string_overlap"], [])

    def test_history_ablation_is_identical_until_second_failure(self):
        failure = TrajStep("send_message(to='x', subject='y')", "err", False, "missing_argument", "send_message")
        self.assertEqual(correction_card([failure], True), correction_card([failure], False))
        self.assertEqual(correction_card([failure, failure], True)["matching_failures"], 2)
        self.assertEqual(correction_card([failure, failure], False)["matching_failures"], 1)
        self.assertIsNone(correction_card([failure, TrajStep("list_files(folder='notes')", "[]", True)], True))

    def test_replay_all_frozen_cases_and_no_world_mutation(self):
        for name in ("smoke", "pilot"):
            cases = read_jsonl(PROJECT / "data" / f"{name}.jsonl")
            for case in cases:
                env = make_env()
                task = Task(**case["task"])
                env.reset(task)
                steps = []
                for action in case["prefix_actions"]:
                    outcome = env.step(task, action)
                    self.assertTrue(outcome.ok)
                    steps.append(trajectory_step(action, outcome))
                before = asdict(env.world(task))
                for _ in range(case["history_depth"]):
                    outcome = env.step(task, case["failed_action"])
                    self.assertFalse(outcome.ok)
                    self.assertEqual(outcome.error_type, case["error_family"])
                    self.assertEqual(before, asdict(env.world(task)))
                    steps.append(trajectory_step(case["failed_action"], outcome))
                messages, card = render(env, task, steps, "positive_history")
                self.assertNotIn(case["failed_action"], "\n".join(m["content"] for m in messages))
                self.assertFalse(literal_leak_audit(steps, card)["full_failed_call_present"])
                for action in case["private_reference_tail"]:
                    self.assertTrue(env.step(task, action).ok)
                self.assertTrue(env.is_solved(task))

    def test_generated_repeat_denominator_excludes_injected_history(self):
        a = TrajStep("read_file(path='no')", "missing", False, "file_not_found", "read_file")
        b = TrajStep("list_files(folder='notes')", "[]", True)
        value = metrics([a, b], [a, a], "controlled", False)
        self.assertEqual(value["next_action_canonical_repeat"], 1)
        self.assertEqual(value["n_generated_failures"], 1)
        self.assertEqual(value["ftb_canonical_repeat_rate"], 1)

    def test_bootstrap_counts_tasks_instead_of_injected_variants(self):
        a = clustered_interval({"task1": [0] * 20, "task2": [1]}, n_boot=200)
        b = clustered_interval({"task1": [0], "task2": [1]}, n_boot=200)
        self.assertEqual(a, b)
        self.assertEqual(a[0], 0.5)

    def test_compatible_api_payload_and_usage_without_network(self):
        backend = ChatAPIBackend({"base_url": "http://localhost:8080/v1", "model": "test-model",
                                  "base_url_env": "PE_TEST_UNUSED_URL", "model_env": "PE_TEST_UNUSED_MODEL",
                                  "send_seed": False, "max_new_tokens": 64})
        body = {"choices": [{"message": {"content": "finish()"}, "finish_reason": "stop"}],
                "model": "test-model-revision", "usage": {"prompt_tokens": 100, "completion_tokens": 3}}
        backend.opener = Mock()
        backend.opener.open.return_value = io.BytesIO(json.dumps(body).encode())
        messages = [{"role": "user", "content": "Do the task"}]
        response = backend.generate(messages, 42, {})
        request = backend.opener.open.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(payload["messages"], messages)
        self.assertNotIn("seed", payload)
        self.assertEqual(payload["max_tokens"], 64)
        self.assertEqual(response["model"], "test-model-revision")
        self.assertEqual(response["completion_tokens"], 3)

    def test_private_solution_is_not_a_feedback_input(self):
        case = read_jsonl(PROJECT / "data" / "smoke.jsonl")[0]
        env = make_env()
        task = Task(**case["task"])
        task.gold_actions = ["PRIVATE_SOLUTION_SENTINEL"]
        env.reset(task)
        failure = trajectory_step(case["failed_action"], env.step(task, case["failed_action"]))
        for condition in ("positive_current", "positive_history", "abstract", "verbatim_positive"):
            messages, _ = render(env, task, [failure], condition)
            self.assertNotIn("PRIVATE_SOLUTION_SENTINEL", str(messages))


if __name__ == "__main__":
    unittest.main()

