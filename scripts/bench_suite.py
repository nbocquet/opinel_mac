"""Run the Opinel MLX benchmark series on this Mac and store the results under benchmarks/.

    .venv-mlx/bin/python scripts/bench_suite.py --machine m5-alice          # full series, about 45 min
    .venv-mlx/bin/python scripts/bench_suite.py --machine m5-alice --quick  # about 10 min

The machine sheet (benchmarks/machines/<id>.json) is written on the first run:
chip, GPU / CPU cores, unified memory, macOS, MLX and mlx-lm versions, OVH and
repository commits, weight format, attention cache layout, power source. No host
name or serial number: the id is a chip and a pseudonym.

Results go to benchmarks/results/<date>/<id>/<bench>.json. An existing result is
never overwritten: a rerun resumes where the previous one stopped. Each result
gets a "meta" block: machine id, date, commit, active OLALA_* variables.

Before starting, the load is checked: battery power, a software inventory scan
(ndtrack, flxfsscan) or a virtual machine above 20 % CPU slow the GPU and the
CPU-bound parts of the speculative loop (see benchmarks/README.md).
"""

import argparse
import datetime
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ROOT / "benchmarks"
PYTHON = sys.executable
MACHINE_ID = re.compile(r"^[a-z][a-z0-9]*-[a-z0-9][a-z0-9-]*$")
BACKGROUND = ("ndtrack", "flxfsscan", "Virtualization", "xprotectd")


def run(command):
    return subprocess.run(command, capture_output=True, text=True, check=False).stdout.strip()


def sysctl(name):
    return run(["sysctl", "-n", name])


def gpu_cores():
    out = run(["system_profiler", "SPDisplaysDataType"])
    match = re.search(r"Total Number of Cores:\s*(\d+)", out)
    return int(match.group(1)) if match else None


def power_source():
    out = run(["pmset", "-g", "batt"])
    return "secteur" if "AC Power" in out else "batterie" if "Battery Power" in out else "inconnue"


def busy_background(ps_output, threshold=20.0):
    """Background processes known to skew the measurements, above threshold % CPU: [(name, cpu)]."""
    found = []
    for line in ps_output.splitlines()[1:]:
        parts = line.strip().split(None, 1)
        if len(parts) != 2:
            continue
        try:
            cpu = float(parts[0].replace(",", "."))
        except ValueError:
            continue
        name = parts[1]
        if cpu >= threshold and any(key in name for key in BACKGROUND):
            found.append((Path(name.split()[0]).name, cpu))
    return found


def package_version(name):
    from importlib.metadata import PackageNotFoundError, version
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def machine_sheet(machine, comment):
    from opinel_mlx import default_folder, validate_provenance

    folder = default_folder()
    upstream = json.loads((ROOT / "vendor" / "ovh_olala" / "UPSTREAM.json").read_text())
    return {
        "id": machine,
        "chip": sysctl("machdep.cpu.brand_string"),
        "gpu_cores": gpu_cores(),
        "cpu_cores": {"total": int(sysctl("hw.ncpu") or 0), "performance": int(sysctl("hw.perflevel0.logicalcpu") or 0),
                      "efficiency": int(sysctl("hw.perflevel1.logicalcpu") or 0)},
        "memory_gb": round(int(sysctl("hw.memsize") or 0) / 2 ** 30),
        "macos": platform.mac_ver()[0],
        "mlx": package_version("mlx"),
        "mlx_lm": package_version("mlx-lm"),
        "ovh_commit": upstream.get("revision"),
        "weights_format": validate_provenance(folder)["format"],
        "attention_cache": os.environ.get("OLALA_TPA_CACHE", "auto"),
        "comment": comment,
    }


def commit():
    return run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"])


def benches(quick):
    """(name, command, output flag): the series, in order."""
    repeats = ["--repeats", "8" if quick else "20"]
    loop = ["--limit", "4", "--tokens", "128"] if quick else []
    series = [
        ("profile_verify", [PYTHON, "scripts/profile_verify.py", *repeats]),
        ("profile_verify_rows", [PYTHON, "scripts/profile_verify.py", "--rows", *repeats]),
        ("speculative_greedy", [PYTHON, "scripts/check_speculative.py", "--steps", "2", "3", "4", *loop]),
        ("speculative_sampled", [PYTHON, "scripts/check_speculative.py", "--steps", "2", "3", "4",
                                 "--temperature", "0.8", "--top-p", "0.9", "--top-k", "40", *loop]),
    ]
    if not quick:
        series.append(("chat_speed_variee", [PYTHON, "scripts/replay_conversation.py", "speed",
                                             "--conversation", "evaluation/conversation_variee.json"]))
    return series


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--machine", required=True, help="Puce et pseudo, par exemple m5-alice (pas de nom d'hôte)")
    parser.add_argument("--quick", action="store_true", help="Série courte, sans la conversation (environ 10 min)")
    parser.add_argument("--comment", default="", help="Commentaire libre pour la fiche machine")
    parser.add_argument("--date", default=datetime.date.today().isoformat())
    parser.add_argument("--force", action="store_true", help="Lancer même si la machine est chargée")
    args = parser.parse_args()
    if not MACHINE_ID.match(args.machine):
        parser.error("--machine : puce puis pseudo en minuscules, par exemple m4-bob.")
    sys.path.insert(0, str(ROOT))

    power = power_source()
    busy = busy_background(run(["ps", "-Ao", "pcpu,comm", "-r"]))
    warnings = ([f"alimentation : {power}"] if power != "secteur" else []) + \
               [f"{name} à {cpu:.0f} % de CPU" for name, cpu in busy]
    if warnings:
        print("Machine chargée : " + " ; ".join(warnings) + ". Les mesures seraient faussées.", file=sys.stderr)
        if not args.force:
            print("Attendre la fin de ces tâches, ou relancer avec --force.", file=sys.stderr)
            return 2

    sheet_path = BENCHMARKS / "machines" / f"{args.machine}.json"
    if not sheet_path.exists():
        sheet_path.parent.mkdir(parents=True, exist_ok=True)
        sheet_path.write_text(json.dumps(machine_sheet(args.machine, args.comment), ensure_ascii=False, indent=2)
                              + "\n")
        print(f"Fiche machine : {sheet_path.relative_to(ROOT)}")
    folder = BENCHMARKS / "results" / args.date / args.machine
    folder.mkdir(parents=True, exist_ok=True)
    for name, command in benches(args.quick):
        output = folder / f"{name}.json"
        if output.exists():
            print(f"Déjà fait : {output.relative_to(ROOT)}")
            continue
        print(f"\n== {name} ==", flush=True)
        partial = output.with_suffix(".partiel.json")
        finished = subprocess.run([*command, "--output", str(partial)], cwd=ROOT, check=False)
        if finished.returncode != 0 or not partial.exists():
            print(f"Échec de {name} (code {finished.returncode}) : relancer reprendra ici.", file=sys.stderr)
            return 1
        result = json.loads(partial.read_text())
        result["meta"] = {"machine": args.machine, "date": args.date, "commit": commit(), "quick": args.quick,
                          "power": power_source(), "olala": {k: v for k, v in os.environ.items()
                                                             if k.startswith("OLALA_")}}
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        partial.unlink()
    print(f"\nRésultats : {folder.relative_to(ROOT)}. Synthèse : .venv-mlx/bin/python scripts/bench_summary.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
