"""Reproducible reliability evaluations, separate from software unit tests."""

from collections import Counter
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from statistics import median
import tempfile
import time
import unicodedata


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def verify_assets(folder, manifest, progress=lambda name: None):
    """Verify the pinned reference's weights, configuration, tokenizer and code."""
    for name, expected in manifest["files"].items():
        path = Path(folder) / name
        if path.stat().st_size != expected["size"]:
            raise ValueError(f"Taille incorrecte : {name}")
        if expected["algorithm"] == "sha256":
            with path.open("rb") as handle:
                actual = hashlib.file_digest(handle, "sha256").hexdigest()
        elif expected["algorithm"] == "git-sha1":
            data = path.read_bytes()
            actual = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        else:
            raise ValueError("Algorithme d'empreinte inconnu.")
        if actual != expected["digest"]:
            raise ValueError(f"Empreinte incorrecte : {name}")
        progress(name)


def load_suite(path):
    raw = Path(path).read_bytes()
    suite = json.loads(raw)
    ids = set()
    for case in suite["cases"]:
        if case["id"] in ids:
            raise ValueError(f"Cas dupliqué : {case['id']}")
        ids.add(case["id"])
        if case["category"] not in suite["categories"] or not case.get("rubric"):
            raise ValueError(f"Grille invalide : {case['id']}")
        messages = case["messages"]
        if not messages or len(messages) % 2 != 1:
            raise ValueError(f"Conversation incomplète : {case['id']}")
        if any(message["role"] != ("user" if index % 2 == 0 else "assistant")
               for index, message in enumerate(messages)):
            raise ValueError(f"Rôles invalides : {case['id']}")
        check = case.get("check")
        if check and (check.get("type") not in ("exact", "json", "bullets", "max_words")
                      or check.get("scope") not in ("full", "format")):
            raise ValueError(f"Vérification invalide : {case['id']}")
    suite["sha256"] = hashlib.sha256(raw).hexdigest()
    return suite


def normalized(text):
    text = unicodedata.normalize("NFKD", text.casefold())
    text = "".join(char for char in text if not unicodedata.combining(char))
    return text.strip().rstrip(".!?").strip()


def check_answer(case, answer):
    check = case.get("check")
    if not check:
        return {"verdict": "unreviewed", "reason": "Lecture de la grille nécessaire."}
    kind, expected = check["type"], check["expected"]
    if kind == "exact":
        passed = normalized(answer) == normalized(expected)
    elif kind == "json":
        try:
            actual = json.loads(answer)
            # JSON booleans compare equal to integers in Python; keep types strict.
            passed = (type(actual) is dict and actual.keys() == expected.keys()
                      and all(type(actual[key]) is type(value) and actual[key] == value
                              for key, value in expected.items()))
        except (ValueError, TypeError):
            passed = False
    elif kind == "bullets":
        lines = answer.strip().splitlines()
        passed = (len(lines) == len(expected) and all(line.startswith("- ") for line in lines)
                  and Counter(normalized(line[2:]) for line in lines)
                  == Counter(normalized(value) for value in expected))
    else:
        passed = 0 < len(answer.split()) <= expected
    verdict = "fail" if not passed else ("pass" if check["scope"] == "full" else "unreviewed")
    return {"verdict": verdict, "type": kind, "scope": check["scope"],
            "check_passed": passed,
            "reason": "Critère déterministe." if verdict != "unreviewed" else
                      "Format respecté ; contenu à examiner."}


def result_verdict(result):
    if result["status"] != "ok":
        return "error"
    review = result.get("review")
    if review:
        return review["verdict"] if review["verdict"] in ("pass", "fail") else "unreviewed"
    return result["automatic"]["verdict"]


def summarize(report):
    by_id = {result["id"]: result for result in report["results"]}
    if (len(by_id) != len(report["results"]) or len(set(report["case_ids"])) != len(report["case_ids"])
            or set(by_id) - set(report["case_ids"])):
        raise ValueError("Résultats dupliqués ou hors sélection.")
    counts = Counter(result_verdict(result) for result in report["results"])
    counts["not_run"] = len(report["case_ids"]) - len(by_id)
    categories = {}
    for result in report["results"]:
        categories.setdefault(result["category"], Counter())[result_verdict(result)] += 1
    total = len(report["case_ids"])
    complete = counts["unreviewed"] == 0 and counts["not_run"] == 0
    reviewers = Counter(result["review"]["kind"] for result in report["results"] if result.get("review"))
    durations = [result["seconds"] for result in report["results"] if result["status"] == "ok"]
    return {"total": total, **{key: counts[key] for key in ("pass", "fail", "error", "unreviewed", "not_run")},
            "score": counts["pass"] / total if complete and total else None,
            "reviewers": dict(reviewers), "by_category": categories,
            "median_seconds_completed": median(durations) if durations else None}


def save_report(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    report["updated_at"] = now()
    report["summary"] = summarize(report)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run_evaluation(suite, config, output, generate, case_ids=None, resume=False, progress=print):
    selected = [case for case in suite["cases"] if case_ids is None or case["id"] in case_ids]
    if not selected or (case_ids is not None and set(case_ids) != {case["id"] for case in selected}):
        raise ValueError("Sélection de cas vide ou inconnue.")
    planned = [case["id"] for case in selected]
    output = Path(output)
    identity = {"suite_sha256": suite["sha256"], "configuration": config, "case_ids": planned}
    if output.exists():
        if not resume:
            raise ValueError("Rapport existant : utiliser --resume ou un nouveau chemin.")
        report = json.loads(output.read_text())
        if any(report.get(key) != value for key, value in identity.items()):
            raise ValueError("Reprise refusée : jeu, sélection ou configuration différents.")
    else:
        if resume:
            raise ValueError("Aucun rapport à reprendre.")
        report = {"schema_version": 1, "suite_name": suite["name"], **identity,
                  "created_at": now(), "results": []}
    save_report(output, report)
    completed = {result["id"] for result in report["results"]}
    suite_positions = {case["id"]: index for index, case in enumerate(suite["cases"])}
    for index, case in enumerate(selected):
        if case["id"] in completed:
            continue
        progress(f"[{index + 1}/{len(selected)}] {case['id']} ({case['category']})")
        started = time.perf_counter()
        result = {"id": case["id"], "category": case["category"], "messages": copy.deepcopy(case["messages"]),
                  "rubric": case["rubric"], "sources": case.get("sources", []),
                  "answer": None, "error": None, "review": None,
                  "generation_seed": config["seed"] + suite_positions[case["id"]]}
        try:
            # Only conversation messages reach the model; never the expected answer or rubric.
            result["answer"] = generate(copy.deepcopy(case["messages"]), result["generation_seed"])
            result["status"] = "ok"
            result["automatic"] = check_answer(case, result["answer"])
        except (ValueError, RuntimeError) as error:
            result.update(status="error", error=str(error), automatic={"verdict": "unreviewed"})
        except KeyboardInterrupt:
            # Save completed cases only; resume will retry the interrupted case.
            save_report(output, report)
            raise
        result["seconds"] = round(time.perf_counter() - started, 3)
        report["results"].append(result)
        save_report(output, report)
        progress(f"  {result_verdict(result)} · {result['seconds']:.1f} s")
    return report


def annotate_report(path, case_id, verdict, note, reviewer, kind):
    if verdict not in ("pass", "fail", "uncertain") or kind not in ("human", "assistant"):
        raise ValueError("Verdict ou type de relecteur invalide.")
    if not note.strip() or not reviewer.strip():
        raise ValueError("La justification et le nom du relecteur sont obligatoires.")
    report = json.loads(Path(path).read_text())
    for result in report["results"]:
        if result["id"] == case_id:
            if result["status"] != "ok":
                raise ValueError("Une erreur de génération ne peut pas être notée comme une réponse.")
            if result.get("review"):
                result.setdefault("review_history", []).append(result["review"])
            result["review"] = {"verdict": verdict, "note": note, "reviewer": reviewer, "kind": kind, "at": now()}
            save_report(path, report)
            return report
    raise ValueError("Cas absent du rapport.")


def compare_reports(reports):
    if len(reports) < 2:
        raise ValueError("Fournir au moins deux rapports.")
    first = reports[0]
    if any(report["suite_sha256"] != first["suite_sha256"] or report["case_ids"] != first["case_ids"]
           for report in reports[1:]):
        raise ValueError("Rapports non comparables : jeu ou sélection différents.")
    configurations = [report["configuration"] for report in reports]
    keys = set().union(*(config.keys() for config in configurations))
    differences = {key: [config.get(key) for config in configurations]
                   for key in sorted(keys)
                   if any(config.get(key) != configurations[0].get(key) for config in configurations[1:])}
    by_ids = [{result["id"]: result for result in report["results"]} for report in reports]
    paired = [{"id": case_id, "verdicts": [result_verdict(results[case_id]) if case_id in results else "not_run"
                                             for results in by_ids]} for case_id in first["case_ids"]]
    return {"suite_sha256": first["suite_sha256"], "configuration_differences": differences,
            "runs": [{"configuration": report["configuration"], "summary": summarize(report)} for report in reports],
            "paired_cases": paired,
            "caution": "Un score partiel ou une configuration différente ne démontre pas une amélioration causale."}
