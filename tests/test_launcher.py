from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from opinel_mac import ROOT


class LauncherTests(unittest.TestCase):
    def test_backend_environment_and_explicit_model_path_are_respected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shutil.copyfile(ROOT / "opinel", root / "opinel")
            for environment in (".venv", ".venv-mlx"):
                python = root / environment / "bin" / "python"
                python.parent.mkdir(parents=True)
                python.write_text('#!/bin/sh\nprintf "%s\\n" "$0" "$@"\n')
                python.chmod(0o755)
            cases = [
                (["chat"], ".venv"),
                (["chat", "--backend", "mlx"], ".venv-mlx"),
                (["chat", "--backend=mlx"], ".venv-mlx"),
                (["chat", "--backend=mlx", "--backend", "torch"], ".venv"),
                (["generate", "--model-dir", "/custom/model"], ".venv"),
            ]
            for arguments, environment in cases:
                result = subprocess.run(["sh", str(root / "opinel"), *arguments],
                                        text=True, capture_output=True, check=True)
                lines = result.stdout.splitlines()
                self.assertEqual(lines[0], str(root / environment / "bin" / "python"))
                self.assertEqual(lines[2:], arguments)
