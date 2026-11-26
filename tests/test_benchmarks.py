import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import bench_suite  # noqa: E402
import bench_summary  # noqa: E402


class BackgroundLoadTests(unittest.TestCase):
    def test_known_background_tasks_above_threshold_are_reported(self):
        ps = """ %CPU COMM
 99.7 /opt/managesoft/libexec/ndtrack
 40.7 /System/Library/Frameworks/Virtualization.framework/Versions/A/XPCServices/com.apple.Virtualization.VirtualMachine
 33,0 /usr/libexec/xprotectd
 12.0 /opt/managesoft/libexec/flxfsscan
 80.0 /Applications/Brave Browser.app/Contents/MacOS/Brave Browser
"""
        names = [name for name, _ in bench_suite.busy_background(ps)]
        self.assertEqual(names, ["ndtrack", "com.apple.Virtualization.VirtualMachine", "xprotectd"])

    def test_machine_id_is_a_chip_and_a_pseudonym(self):
        self.assertTrue(bench_suite.MACHINE_ID.match("m4-bob"))
        self.assertFalse(bench_suite.MACHINE_ID.match("MacBook-Pro-de-Bob.local"))
        self.assertFalse(bench_suite.MACHINE_ID.match("m5"))


class SummaryTests(unittest.TestCase):
    def test_summary_reads_each_bench_and_marks_missing_ones(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "machines").mkdir()
            (root / "machines" / "m5-test.json").write_text(json.dumps({"chip": "Apple M5"}))
            folder = root / "results" / "2026-11-26" / "m5-test"
            folder.mkdir(parents=True)
            (folder / "profile_verify.json").write_text(json.dumps(
                {"milliseconds": {"4 (noyaux de décodage, opinel_verify)": 21.6}}))
            (folder / "speculative_greedy.json").write_text(json.dumps(
                {"plain_tokens_per_second": 77.3, "speculative": {"3": {"speedup": 1.21}}}))
            (folder / "chat_speed_variee.json").write_text(json.dumps(
                {"tokens_per_second": {"normal": 58.0, "spéculatif": 63.2}}))
            text = bench_summary.summary(root)
        line = text.strip().splitlines()[-1]
        self.assertEqual(line, "| m5-test | 2026-11-26 | Apple M5 | 77,3 tokens/s | 21,6 ms | ×1,21 | — | ×1,09 |")


if __name__ == "__main__":
    unittest.main()
