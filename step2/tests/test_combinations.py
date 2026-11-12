from pathlib import Path
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import unittest
import os
from tokenizers import Tokenizer

from pe_feedback.common import PE, read_rows
from pe_feedback.experiment import replay
from pe_feedback.feedback import FeedbackEngine, compose
from pinkelephant.feedback import correction_card, render_card


class CombinationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = Tokenizer.from_file(str(Path(os.environ.get("MODEL_DIR", "C:/Projects/AI/models")) / "Qwen2.5-0.5B-Instruct" / "tokenizer.json"))

    def test_fact_block_fixed_and_original_guidance_unchanged(self):
        measure = lambda text: len(self.tokenizer.encode(text, add_special_tokens=False).ids)
        for case in read_rows(PE / "step1" / "data" / "pilot.jsonl"):
            env, task, steps = replay(case)
            cards = {}
            for mode in ("facts_control", "facts_retry", "facts_diagnosis", "facts_positive"):
                msgs, card, _ = compose(env.system_prompt(task), task.goal, steps, mode, FeedbackEngine(), measure, 384)
                self.assertEqual(msgs[-1]["content"], card["text"])
                cards[mode] = card
            self.assertEqual(len({c["facts_text"] for c in cards.values()}), 1)
            legacy = correction_card(steps, True)
            self.assertEqual(cards["facts_positive"]["addition"], render_card(legacy))
            self.assertEqual(cards["facts_diagnosis"]["addition"], render_card(legacy).split("\n", 1)[0])
            self.assertEqual(cards["facts_retry"]["matching_count"], legacy["matching_failures"])

    def test_no_new_card_after_success(self):
        case = read_rows(PE / "step1" / "data" / "pilot.jsonl")[0]
        env, task, steps = replay(case)
        from pe_feedback.bridge import trajectory_step
        steps.append(trajectory_step("list_files(folder='notes')", env.step(task, "list_files(folder='notes')")))
        _, card, _ = compose(env.system_prompt(task), task.goal, steps, "facts_positive", FeedbackEngine())
        self.assertIsNone(card)


if __name__ == "__main__":
    unittest.main()
