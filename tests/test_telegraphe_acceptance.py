import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class HeadVocabularyTests(unittest.TestCase):
    def test_invalid_sizes_are_rejected_before_loading_the_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            ranking = Path(tmp) / "vocab.json"
            ranking.write_text(json.dumps({"ranked": [7, 3, 9, 1]}))
            for count in (5, -1):
                with self.subTest(count=count):
                    result = subprocess.run(
                        [sys.executable, str(ROOT / "scripts" / "telegraphe_acceptance.py"),
                         "--vocab-file", str(ranking), "--head-vocab", "0", "4", str(count),
                         "--model-dir", str(Path(tmp) / "missing-model")],
                        capture_output=True, text=True,
                    )
                    self.assertEqual(result.returncode, 2, result.stderr)
                    self.assertIn(f"{count} hors limites", result.stderr)
                    self.assertIn("ne classe que 4 tokens", result.stderr)
                    self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
