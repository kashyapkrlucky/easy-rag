"""Evaluation reports: a JSON file and a human-readable summary."""

import json
from pathlib import Path
from typing import Any


def write_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _pct(value: float | None) -> str:
    return "  n/a" if value is None else f"{value * 100:5.1f}%"


def _secs(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f}s"


def format_summary(report: dict[str, Any]) -> str:
    dataset, config = report["dataset"], report["config"]
    lines = [
        f"Dataset {dataset['name']}: {dataset['questions']} questions {dataset['kinds']}",
        f"Models: {config['embedding_model']} | {config['llm_model']} | "
        f"prompt v{config['prompt_version']} | mode={config['retrieval_mode']} "
        f"rerank={config['rerank_enabled']} policy={config['rag_ungrounded_policy']}",
        "",
        f"Retrieval (answerable questions, top_k={config['retrieval_top_k']}):",
        f"  {'config':16} {'hit@1':>7} {'hit@k':>7} {'MRR':>6}",
    ]
    for name, metrics in report["retrieval"].items():
        m = metrics["all"]
        lines.append(
            f"  {name:16} {_pct(m['hit_at_1']):>7} {_pct(m['hit_at_k']):>7} {m['mrr']:6.3f}"
        )

    calibration = report["calibration"]
    scores, suggestion = calibration["vector_top1_score"], calibration["suggested_min_score"]
    lines += [
        "",
        "Calibration (top-1 vector score):",
        f"  answerable   {scores['answerable']}",
        f"  unanswerable {scores['unanswerable']}",
    ]
    if suggestion:
        lines.append(
            f"  suggested min_score {suggestion['threshold']} "
            f"(balanced accuracy {_pct(suggestion['balanced_accuracy'])}; would reject "
            f"{suggestion['answerable_rejected']} answerable, admit "
            f"{suggestion['unanswerable_accepted']} unanswerable)"
        )

    answers = report.get("answers")
    if answers:
        a, u, i = answers["answerable"], answers["unanswerable"], answers["injection"]
        latency = answers["latency_seconds"]["total"]
        lines += [
            "",
            "Answers (full pipeline):",
            f"  answerable   correct {_pct(a['correct_rate'])}  answered {_pct(a['answered_rate'])}"
            f"  fact recall {_pct(a['fact_recall_mean'])}"
            f"  citation precision {_pct(a['citation_precision_mean'])}"
            f"  attribution {a['attribution']}",
            f"  unanswerable refusal accuracy {_pct(u['refusal_accuracy'])} ({u['count']})",
            f"  injection    resisted {_pct(i['resisted_rate'])}"
            f"  correct {_pct(i['correct_rate'])}",
            "  by kind      "
            + "  ".join(f"{k} {_pct(v['correct_rate'])}" for k, v in answers["by_kind"].items()),
            f"  latency      p50 {_secs(latency['p50'])}  p95 {_secs(latency['p95'])}",
        ]
        if answers["errors"]:
            lines.append(f"  errors       {len(answers['errors'])}: {answers['errors'][:3]}")
    return "\n".join(lines)
