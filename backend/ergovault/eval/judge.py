"""Судья: опирается ли одно утверждение на один фрагмент (T-30, шаг 1).

Вход судьи — ровно одно утверждение и ровно один фрагмент, выход — JSON с `verdict`
(`supported` | `partially` | `not_supported`) и краткой причиной. Судья оценивает только
наличие опоры: ни стиль, ни полезность, ни полнота ответа его не касаются.

Судья — измерительный прибор, и он сам подлежит проверке. Прибор без калибровки
показаний не публикует: `calibrate()` сравнивает его вердикты с ручной разметкой
(доля совпадений и Cohen's kappa), с другой моделью и другим промптом, а разброс
считает по повторам. Ручную разметку делает человек; модель не проставляет её за него —
поэтому пары с `human_verdict: null` в согласие не входят, а не «засчитываются» сами себе.
"""

from __future__ import annotations

import itertools
import re
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import psycopg
import yaml
from pydantic import BaseModel, ValidationError

from ergovault import answering, db
from ergovault.llm import LLMProvider, get_provider
from ergovault.models import AnswerPayload, Hit
from ergovault.settings import REPO_ROOT, Settings, get_settings

Verdict = Literal["supported", "partially", "not_supported"]
VERDICTS: tuple[Verdict, ...] = ("supported", "partially", "not_supported")

DEFAULT_JUDGE_PROMPT = "j1"
ALTERNATE_JUDGE_PROMPT = "j2"
JUDGE_MAX_TOKENS = 500
# Лежит рядом с eval/golden, а не внутри: load_golden_set() читает оттуда все *.yml
# как проверочные вопросы, и файл пар сломал бы загрузку набора.
PAIRS_PATH = REPO_ROOT / "eval" / "judge-pairs.yml"
MIN_CALIBRATION_PAIRS = 40  # тикет T-30: 40–50 размеченных вручную пар


class JudgeError(RuntimeError):
    """Ответ судьи не разобран. Не вердикт: неизвестное не превращается в `not_supported`."""


# --- утверждения ответа ---------------------------------------------------------------


class Claim(BaseModel):
    """Одно утверждение ответа и номера фрагментов, на которые оно ссылается."""

    text: str  # без ссылок [n]: судья не должен видеть разметку ответа
    refs: list[int]


_MIN_CLAIM_CHARS = 20
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([.,;:!?»)])")


def _tidy(text: str) -> str:
    """Убрать следы снятых ссылок: двойные пробелы, пробел перед знаком, маркеры списка."""
    return _SPACE_BEFORE_PUNCT.sub(r"\1", " ".join(text.split())).strip(" \t-*·—#").strip()


def split_claims(answer_markdown: str) -> list[Claim]:
    """Предложения ответа, у которых есть хотя бы одна ссылка [n].

    Утверждение без ссылки проверять не по чему: у него нет заявленной опоры, и его
    судят не здесь, а проверка ответа в `answering.build_payload` (ответ без единой
    ссылки допустим только при отказе). Короткие обрывки (заголовки, «Итого:»)
    отбрасываются: они не несут проверяемого содержания.
    """
    claims: list[Claim] = []
    for part in answering.SENTENCE_SPLIT.split(answer_markdown):
        refs = [int(n) for n in answering.REF.findall(part)]
        if not refs:
            continue
        text = _tidy(answering.REF.sub("", part))
        if len(text) >= _MIN_CLAIM_CHARS:
            claims.append(Claim(text=text, refs=sorted(dict.fromkeys(refs))))
    return claims


# --- один вердикт ---------------------------------------------------------------------


class JudgeConfig(BaseModel):
    """Настройка судьи: модель и версия промпта. Обе пишутся в каждую строку прогона."""

    model: str
    prompt_version: str = DEFAULT_JUDGE_PROMPT

    @property
    def name(self) -> str:
        return f"{self.prompt_version}@{self.model}"


class JudgeCall(BaseModel):
    """Один вызов судьи. Хранится сырым: пересчитать метрики можно, не вызывая модель."""

    pair_id: str
    config: str
    model: str
    prompt_version: str
    prompt_hash: str
    repeat: int
    verdict: Verdict
    reason: str
    latency_ms: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    raw_cost_usd: float | None


class _JudgeAnswer(BaseModel):
    verdict: Verdict
    reason: str = ""


def judge_pair(
    claim: str,
    fragment: str,
    *,
    config: JudgeConfig,
    pair_id: str = "",
    repeat: int = 0,
    provider: LLMProvider | None = None,
) -> JudgeCall:
    """Вердикт по одной паре «утверждение — фрагмент»."""
    system, user_template, prompt_hash = answering.load_prompt(config.prompt_version)
    user = user_template.substitute(claim=claim.strip(), fragment=fragment.strip())
    response = (provider or get_provider()).complete(
        system=system,
        user=user,
        model=config.model,
        cache_system=True,
        max_tokens=JUDGE_MAX_TOKENS,
    )
    try:
        parsed = _JudgeAnswer.model_validate_json(answering.strip_code_fence(response.text))
    except ValidationError as exc:
        raise JudgeError(f"судья вернул не JSON по схеме: {response.text[:300]!r}") from exc

    return JudgeCall(
        pair_id=pair_id,
        config=config.name,
        model=response.model,
        prompt_version=config.prompt_version,
        prompt_hash=prompt_hash,
        repeat=repeat,
        verdict=parsed.verdict,
        reason=parsed.reason.strip(),
        latency_ms=response.latency_ms,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        cost_usd=response.billed_cost_usd,
        raw_cost_usd=response.raw_cost_usd,
    )


# --- подтверждённость ответа ------------------------------------------------------------


class ClaimVerdict(BaseModel):
    claim: str
    refs: list[int]
    verdict: Verdict  # лучший из вердиктов по процитированным фрагментам
    per_fragment: dict[str, Verdict]  # chunk_id → вердикт


_RANK = {"supported": 2, "partially": 1, "not_supported": 0}


def judge_answer(
    payload: AnswerPayload,
    hits: Sequence[Hit],
    *,
    config: JudgeConfig,
    provider: LLMProvider | None = None,
) -> list[ClaimVerdict]:
    """Вердикт по каждому утверждению ответа: опирается ли оно на то, на что ссылается.

    Утверждение со ссылками на несколько фрагментов получает лучший из их вердиктов:
    оно и заявлено как опирающееся на любой из них. Это подтверждённость текста, а не
    независимость источников — независимость считается по метаданным (CONTRACTS.md §7).
    """
    results: list[ClaimVerdict] = []
    for claim in split_claims(payload.answer_markdown):
        per_fragment: dict[str, Verdict] = {}
        for n in claim.refs:
            if not 1 <= n <= len(hits):
                continue
            hit = hits[n - 1]
            call = judge_pair(claim.text, hit.chunk.text, config=config, provider=provider)
            per_fragment[hit.chunk.id] = call.verdict
        if not per_fragment:
            continue
        best = max(per_fragment.values(), key=lambda v: _RANK[v])
        results.append(
            ClaimVerdict(claim=claim.text, refs=claim.refs, verdict=best, per_fragment=per_fragment)
        )
    return results


# --- калибровка -------------------------------------------------------------------------


class JudgePair(BaseModel):
    """Пара для калибровки. `human_verdict: null` — ещё не размечена человеком.

    Текст фрагмента здесь не хранится: он берётся из БД по `chunk_id`, чтобы разметка
    не разошлась с корпусом при его перезагрузке.
    """

    id: str
    claim: str
    chunk_id: str
    origin: str = ""  # вопрос и вариант, из ответа которого взято утверждение
    human_verdict: Verdict | None = None
    note: str = ""


def load_judge_pairs(path: Path = PAIRS_PATH) -> list[JudgePair]:
    if not path.is_file():
        raise FileNotFoundError(
            f"нет файла пар для калибровки {path}; собрать: ev eval judge-pairs"
        )
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    pairs = [JudgePair(**item) for item in raw]
    ids = Counter(p.id for p in pairs)
    if duplicates := [i for i, c in ids.items() if c > 1]:
        raise ValueError(f"дублирующиеся id пар: {', '.join(sorted(duplicates))}")
    return pairs


def save_judge_pairs(pairs: Sequence[JudgePair], path: Path = PAIRS_PATH) -> None:
    """Записать пары, сохранив уже проставленную человеком разметку нетронутой."""
    path.parent.mkdir(parents=True, exist_ok=True)
    body = yaml.safe_dump(
        [p.model_dump() for p in pairs], allow_unicode=True, sort_keys=False, width=100
    )
    path.write_text(body, encoding="utf-8")


def fragment_text(conn: psycopg.Connection, chunk_id: str) -> str:
    chunk = db.get_chunk(conn, chunk_id)
    if chunk is None:
        raise ValueError(f"чанк {chunk_id} не найден: разметка разошлась с корпусом")
    return chunk.text


def cohen_kappa(
    a: Sequence[str], b: Sequence[str], labels: Sequence[str] = VERDICTS
) -> float | None:
    """Cohen's kappa для двух разметок одного набора. None — согласие не определено.

    None возвращается, когда ожидаемое согласие равно 1 (обе разметки поставили одну
    и ту же метку везде): kappa в этой точке не определена, и подставлять 0 или 1
    значило бы выдать вырожденный случай за измерение.
    """
    if len(a) != len(b):
        raise ValueError("разметки разной длины")
    n = len(a)
    if n == 0:
        return None
    observed = sum(1 for x, y in zip(a, b, strict=True) if x == y) / n
    expected = sum((list(a).count(x) / n) * (list(b).count(x) / n) for x in labels)
    if expected >= 1.0:
        return None
    return (observed - expected) / (1 - expected)


class AgreementMetrics(BaseModel):
    n: int
    observed_agreement: float | None
    cohen_kappa: float | None
    confusion: dict[str, int] = {}  # "человек→судья" → сколько раз


def agreement(human: Sequence[str], judge: Sequence[str]) -> AgreementMetrics:
    if not human:
        return AgreementMetrics(n=0, observed_agreement=None, cohen_kappa=None)
    observed = sum(1 for x, y in zip(human, judge, strict=True) if x == y) / len(human)
    confusion = Counter(f"{x}→{y}" for x, y in zip(human, judge, strict=True))
    return AgreementMetrics(
        n=len(human),
        observed_agreement=observed,
        cohen_kappa=cohen_kappa(human, judge),
        confusion=dict(confusion),
    )


class ConfigCalibration(BaseModel):
    config: str
    repeats: int
    n_pairs: int
    n_labeled: int  # сколько пар размечено человеком — знаменатель согласия
    self_consistency: float | None  # доля пар, где все повторы дали один вердикт
    unstable_pairs: list[str] = []
    vs_human: AgreementMetrics | None  # None — ручной разметки нет, согласие не измерено


class Calibration(BaseModel):
    """Итог калибровки. Всё, что не измерено, остаётся None и так и называется."""

    generated_at: str
    judge_configs: list[ConfigCalibration]
    between_configs: AgreementMetrics | None  # согласие основного судьи с альтернативным
    labeled_enough: bool  # хватает ли ручной разметки по требованию тикета (≥40 пар)
    note: str
    # Ответы, не разобранные по схеме: вердиктом они не становятся (в частности,
    # не превращаются в not_supported), но и молчать о них нельзя.
    unparsed_calls: int = 0


def _majority(verdicts: Sequence[str]) -> str:
    """Мода вердиктов; при равенстве — более осторожный (not_supported < partially)."""
    counts = Counter(verdicts)
    top = max(counts.values())
    return min((v for v, c in counts.items() if c == top), key=lambda v: _RANK[v])


def calibrate(
    pairs: Sequence[JudgePair],
    calls: Sequence[JudgeCall],
    *,
    generated_at: str,
    unparsed_calls: int = 0,
) -> Calibration:
    """Свести сырые вызовы судьи в согласие с человеком, между конфигурациями и разброс."""
    labeled = {p.id: p.human_verdict for p in pairs if p.human_verdict is not None}
    by_config: dict[str, list[JudgeCall]] = {}
    for call in calls:
        by_config.setdefault(call.config, []).append(call)

    majority_by_config: dict[str, dict[str, str]] = {}
    configs: list[ConfigCalibration] = []
    for name, group in by_config.items():
        by_pair: dict[str, list[JudgeCall]] = {}
        for call in group:
            by_pair.setdefault(call.pair_id, []).append(call)
        repeats = max(len(v) for v in by_pair.values()) if by_pair else 0
        stable = [pid for pid, cs in by_pair.items() if len({c.verdict for c in cs}) == 1]
        unstable = sorted(set(by_pair) - set(stable))
        majority = {pid: _majority([c.verdict for c in cs]) for pid, cs in by_pair.items()}
        majority_by_config[name] = majority

        common = [pid for pid in majority if pid in labeled]
        vs_human = (
            agreement([labeled[pid] for pid in common], [majority[pid] for pid in common])
            if common
            else None
        )
        configs.append(
            ConfigCalibration(
                config=name,
                repeats=repeats,
                n_pairs=len(by_pair),
                n_labeled=len(common),
                self_consistency=(len(stable) / len(by_pair)) if by_pair else None,
                unstable_pairs=unstable,
                vs_human=vs_human,
            )
        )

    between = None
    if len(majority_by_config) >= 2:
        first, second = list(majority_by_config)[:2]
        shared = sorted(set(majority_by_config[first]) & set(majority_by_config[second]))
        if shared:
            between = agreement(
                [majority_by_config[first][p] for p in shared],
                [majority_by_config[second][p] for p in shared],
            )

    labeled_enough = len(labeled) >= MIN_CALIBRATION_PAIRS
    note = (
        "Судья откалиброван по ручной разметке."
        if labeled_enough
        else (
            f"Ручной разметки {len(labeled)} пар из требуемых {MIN_CALIBRATION_PAIRS}: "
            "согласие судьи с человеком не измерено, подтверждённость публиковать нельзя."
        )
    )
    if unparsed_calls:
        note += (
            f" Ответов судьи не по схеме: {unparsed_calls} — они исключены из подсчёта, "
            "а не засчитаны как «нет опоры»."
        )
    return Calibration(
        generated_at=generated_at,
        judge_configs=sorted(configs, key=lambda c: c.config),
        between_configs=between,
        labeled_enough=labeled_enough,
        note=note,
        unparsed_calls=unparsed_calls,
    )


def run_calibration_calls(
    conn: psycopg.Connection,
    pairs: Sequence[JudgePair],
    *,
    configs: Sequence[JudgeConfig],
    repeats: int,
    provider: LLMProvider | None = None,
    progress=None,
) -> tuple[list[JudgeCall], list[str]]:
    """Прогнать судью по всем парам: каждая конфигурация, каждый повтор.

    Возвращает вердикты и список неразобранных ответов. Ответ не по схеме не становится
    вердиктом: судья, придумавший четвёртое значение, — это отсутствие измерения на
    этой паре, а не «нет опоры».
    """
    calls: list[JudgeCall] = []
    unparsed: list[str] = []
    plan = list(itertools.product(pairs, configs, range(1, repeats + 1)))
    total = len(plan)
    for done, (pair, config, repeat) in enumerate(plan, start=1):
        text = fragment_text(conn, pair.chunk_id)
        started = time.monotonic()
        try:
            call = judge_pair(
                pair.claim,
                text,
                config=config,
                pair_id=pair.id,
                repeat=repeat,
                provider=provider,
            )
            calls.append(call)
            outcome = call.verdict
        except JudgeError as exc:
            unparsed.append(f"{pair.id} {config.name} #{repeat}: {exc}")
            outcome = "ответ не по схеме"
        if progress is not None:
            progress(
                done,
                total,
                f"{pair.id} {config.name} #{repeat} → {outcome}",
                time.monotonic() - started,
            )
    return calls, unparsed


CALIBRATION_PATH = REPO_ROOT / "eval" / "reports" / "judge-calibration.json"
CALLS_PATH = REPO_ROOT / "eval" / "runs" / "judge-calls.jsonl"


def save_calibration(calibration: Calibration, path: Path = CALIBRATION_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(calibration.model_dump_json(indent=2) + "\n", encoding="utf-8")


def load_calibration(path: Path = CALIBRATION_PATH) -> Calibration | None:
    """Результат калибровки, если она проводилась. Нет файла — судья не проверялся."""
    if not path.is_file():
        return None
    return Calibration.model_validate_json(path.read_text(encoding="utf-8"))


def save_calls(calls: Sequence[JudgeCall], path: Path = CALLS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for call in calls:
            stream.write(call.model_dump_json() + "\n")


def load_calls(path: Path = CALLS_PATH) -> list[JudgeCall]:
    if not path.is_file():
        return []
    return [
        JudgeCall.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def default_configs(settings: Settings | None = None) -> list[JudgeConfig]:
    """Основной судья и альтернативный: другая модель И другой промпт (требование T-30)."""
    settings = settings or get_settings()
    return [
        JudgeConfig(model=settings.judge_model, prompt_version=DEFAULT_JUDGE_PROMPT),
        JudgeConfig(model=settings.gen_model, prompt_version=ALTERNATE_JUDGE_PROMPT),
    ]


def run_calibration(
    *,
    repeats: int = 3,
    configs: Sequence[JudgeConfig] | None = None,
    settings: Settings | None = None,
    echo=print,
) -> Calibration:
    """Прогнать судью по парам и записать калибровку. Модель вызывается здесь.

    Прогон делается и без ручной разметки: устойчивость повторов и согласие двух
    конфигураций измеримы сами по себе. Согласие с человеком при этом останется
    неизмеренным, и `labeled_enough` честно скажет, что подтверждённость публиковать рано.
    """
    import datetime

    settings = settings or get_settings()
    configs = list(configs or default_configs(settings))
    pairs = load_judge_pairs()
    if not pairs:
        raise ValueError("файл пар пуст — собрать: ev eval judge-pairs")

    total = len(pairs) * len(configs) * repeats
    echo(f"пар {len(pairs)} × конфигураций {len(configs)} × повторов {repeats} = {total} вызовов")

    def progress(done: int, total_calls: int, what: str, seconds: float) -> None:
        echo(f"[{done}/{total_calls}] {what} — {seconds:.1f} с")

    with psycopg.connect(settings.database_url) as conn:
        calls, unparsed = run_calibration_calls(
            conn, pairs, configs=configs, repeats=repeats, progress=progress
        )

    stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
    calibration = calibrate(pairs, calls, generated_at=stamp, unparsed_calls=len(unparsed))
    for line in unparsed:
        echo(f"не разобран: {line}")
    save_calls(calls)
    save_calibration(calibration)
    echo("")
    echo(calibration.note)
    for cfg in calibration.judge_configs:
        vs = cfg.vs_human
        agreement_text = (
            f"согласие с человеком {vs.observed_agreement:.2f}, kappa "
            f"{vs.cohen_kappa:.2f}" if vs and vs.cohen_kappa is not None else "с человеком не сверялся"
        )
        echo(
            f"  {cfg.config}: устойчивость {cfg.self_consistency:.2f} на {cfg.n_pairs} парах, "
            f"{agreement_text}"
        )
    echo(f"калибровка: {CALIBRATION_PATH.relative_to(REPO_ROOT)}")
    return calibration
