"""Оценка поиска: вопросы → тот же search() → метрики и файлы отчёта."""

import datetime
from functools import partial

from ergovault.search import MODES, search


def evaluate(
    *, mode="hybrid", k=5, include_holdout=False, profile="public", label=None, settings=None
):
    """Провести оценку; holdout открывается только по явному флагу."""
    import psycopg

    from ergovault.eval.dataset import load_golden_set
    from ergovault.eval.metrics import compute_metrics, render_report_markdown, render_stdout_table
    from ergovault.eval.runner import run, save_run_jsonl
    from ergovault.settings import REPO_ROOT, get_settings

    if mode not in MODES:
        raise ValueError(f"--mode должен быть один из {MODES}")

    settings = settings or get_settings()
    questions = load_golden_set(include_holdout=include_holdout)
    if not questions:
        raise ValueError("golden set пуст — нечего прогонять")

    with psycopg.connect(settings.database_url) as conn:
        find = partial(search, conn, mode=mode, settings=settings)
        rows = run(questions, find, mode=mode, k=k, profile=profile, settings=settings)

    report = compute_metrics(rows, k=k)

    ts = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
    run_label = label or mode
    runs_dir = REPO_ROOT / "eval" / "runs"
    reports_dir = REPO_ROOT / "eval" / "reports"
    run_path = runs_dir / f"{ts}-{run_label}.jsonl"
    report_path = reports_dir / f"{ts}-{run_label}.md"

    save_run_jsonl(rows, run_path)
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        render_report_markdown(
            report,
            mode=mode,
            include_holdout=include_holdout,
            corpus_version=settings.corpus_version,
            prompt_version=settings.prompt_version,
            label=run_label,
            run_path=str(run_path.relative_to(REPO_ROOT)),
        ),
        encoding="utf-8",
    )

    print(render_stdout_table(report, mode=mode, include_holdout=include_holdout))
    print()
    print(f"прогон: {run_path.relative_to(REPO_ROOT)}")
    print(f"отчёт:  {report_path.relative_to(REPO_ROOT)}")
