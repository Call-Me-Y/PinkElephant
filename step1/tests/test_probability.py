from pathlib import Path
import importlib.util
import math
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pinkelephant.common import PROJECT, read_jsonl

spec = importlib.util.spec_from_file_location("probability_probe", PROJECT / "scripts" / "probe_probability.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class ProbabilityTests(unittest.TestCase):
    def test_score_excludes_prefix_and_aligns_next_token(self):
        import torch

        class ToyModel:
            def __call__(self, input_ids, use_cache):
                # Each position predicts the actual next token with known odds.
                logits = torch.zeros(1, input_ids.shape[1], 5)
                for i in range(input_ids.shape[1] - 1):
                    logits[0, i, input_ids[0, i + 1]] = 2
                return SimpleNamespace(logits=logits)

        result = probe.score_tokens(ToyModel(), [0, 1, 2], [3, 4], "cpu")
        expected = 2 * (2 - math.log(math.exp(2) + 4))
        self.assertAlmostEqual(result["logp"], expected, places=5)
        self.assertEqual(result["n_tokens"], 2)
        self.assertTrue(result["all_candidate_tokens_greedy"])

    def test_all_pilot_cases_have_same_four_candidates_for_all_conditions(self):
        for case in read_jsonl(PROJECT / "data" / "pilot.jsonl"):
            candidates = probe.candidates_for(case, 4)
            self.assertEqual(len(set(candidates)), 4)
            self.assertEqual(candidates[0], case["failed_action"])
            self.assertEqual(candidates[1], case["private_reference_tail"][0])


if __name__ == "__main__":
    unittest.main()
