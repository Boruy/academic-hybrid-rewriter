import tempfile
import unittest
from pathlib import Path

from academic_rewriter.lexical import LexicalEdit, LexicalFormalizer, evaluate_lexical


class LexicalFormalizerTests(unittest.TestCase):
    def setUp(self):
        self.model = LexicalFormalizer.fit([
            {"source": "The analysts get access to the archive.",
             "target": "The analysts obtain access to the archive."},
            {"source": "The students get access to the logs.",
             "target": "The students obtain access to the logs."},
            {"source": "The interns get credit for their work.",
             "target": "The interns gain credit for their work."},
            {"source": "The annotators get credit for this experiment.",
             "target": "The annotators gain credit for this experiment."},
            {"source": "The threads get stuck during startup.",
             "target": "The threads get stuck during startup.", "focus": "get"},
        ])

    def test_replacement_depends_on_context_and_preserves_other_characters(self):
        source = "Get access to the logs (n=24), then get credit for the analysis."
        result, edits = self.model.rewrite(source)
        second = source.index("get")
        self.assertEqual(result, "Obtain access to the logs (n=24), then gain credit for the analysis.")
        self.assertEqual(edits, [LexicalEdit(0, 3, "Get", "Obtain"),
                                 LexicalEdit(second, second + 3, "get", "gain")])
        source = "See `get access to the logs` or the get-access and GET access endpoints."
        self.assertEqual(self.model.rewrite(source), (source, []))

    def test_unsupported_and_conflicting_contexts_abstain(self):
        for source in ("The threads get stuck during recovery.", "They get upset.", "Get"):
            with self.subTest(source=source):
                self.assertEqual(self.model.rewrite(source), (source, []))

        model = LexicalFormalizer.fit([
            {"source": "The program includes a quick check of the input.",
             "target": "The program includes a rapid check of the input."},
            {"source": "The protocol requires a quick check for errors.",
             "target": "The protocol requires a brief check for errors."},
        ])
        source = "We use a quick check before recording."
        self.assertEqual(model.rewrite(source), (source, []))

    def test_rejects_multiword_and_unrelated_changes(self):
        for target in ("We gain access to 50 logs.", "We gain secure access to 15 logs.",
                       "We  gain access to 15 logs."):
            with self.subTest(target=target), self.assertRaises(ValueError):
                LexicalFormalizer.fit([{"source": "We get access to 15 logs.", "target": target}])

    def test_save_reload_and_held_out_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            self.model.save(path)
            with self.assertRaises(FileExistsError):
                self.model.save(path)
            model = LexicalFormalizer.load(path)
            result = evaluate_lexical(model, [
                {"source": "They will get access after approval.",
                 "target": "They will obtain access after approval."},
                {"source": "The threads get stuck during recovery.",
                 "target": "The threads get stuck during recovery.", "focus": "get"},
            ])
        self.assertEqual(result["focused_replacement_accuracy"], 1.0)
        self.assertEqual(result["exact_sentence_accuracy"], 1.0)
        self.assertEqual(result["unnecessary_edit_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
