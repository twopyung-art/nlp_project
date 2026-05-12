"""Production TC seq 구조에서 5종 패턴을 분류·추출.

분류 규칙 (seq 구조 + procedure 발화 분석 기반):
- Atomic       : step_count == 1
- Confirmation : step == 2 이고 step2 procedure가 짧은 응답어 ("Yes"/"No"/"Run it"/"Cancel"/"Never mind"/"하지마"/"실행"/"Sí"/"No, gracias" 등)
- Branching    : 같은 Title 가진 TC ≥ 2 이고 모두 step ≥ 2 (각 TC가 Step1 이후 분기)
- Compound     : 한 step procedure 내에 다중 도메인 발화 (and/&/, +) 또는 module_sub == "멀티인탠트"/"연속대화 멀티턴"
- Variant      : 그 외 step ≥ 2 (step 간 의존성 없는 평행 발화)

다국어 변형:
- "스페인어(es-mx)" suffix 가진 module_sub 는 동일 카테고리의 다국어 변형으로 함께 집계.

산출물 (output/patterns/):
- pattern_classified.parquet     : per-TC 분류 결과 (raw)
- pattern_library.json           : 패턴별 통계 + 대표 예시
- pattern_distribution.csv       : module × pattern 분포 매트릭스
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DEFAULT_INPUT = Path("/home/lgai/TestCase-Agent/testcase-db/TestCase_US_4771ea.xlsx")
DEFAULT_OUTDIR = Path("/home/lgai/massive_datasets/output/patterns")

# ---------------------------------------------------------------------------
# Heuristic regexes
# ---------------------------------------------------------------------------
RESPONSE_TOKENS = [
    # 영어
    r"\byes\b", r"\bno\b", r"\bokay\b", r"\bok\b", r"\bsure\b",
    r"run it", r"do it", r"cancel", r"never mind", r"stop",
    r"i don'?t want", r"go ahead", r"please do", r"yeah",
    # 한국어
    r"실행", r"해줘", r"하지마", r"취소", r"아니", r"네\b", r"응\b", r"맞아",
    # 스페인어
    r"\bs[ií]\b", r"\bno,? gracias\b", r"adelante", r"ejec[uú]tar",
]
RESPONSE_RE = re.compile("|".join(RESPONSE_TOKENS), re.IGNORECASE)

COMPOUND_CONNECTORS = [
    r"\band\b",
    r"\s&\s",
    r"\s\+\s",
    r"->|→",
    r",\s*and",
    r"해주고",
    r"하고\s",
    r"및\s",
    r"y\s+también",
]
COMPOUND_RE = re.compile("|".join(COMPOUND_CONNECTORS), re.IGNORECASE)

EXPLICIT_COMPOUND_MODULES = {"멀티인탠트", "연속대화 멀티턴"}

LANG_SUFFIXES = ["-스페인어(es-mx)", "-스페인어(es-ma)", "-스페인어(es-ar)",
                 "스페인어(es-mx)", "스페인어(es-ma)"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def strip_lang_suffix(s: Optional[str]) -> str:
    if pd.isna(s):
        return ""
    text = str(s)
    for suffix in LANG_SUFFIXES:
        if suffix in text:
            text = text.replace(suffix, "")
    return text.strip(" -") or "es-mx-untyped"


def extract_utterance(procedure: str) -> str:
    """Procedure 셀에서 첫 dash 라인 발화 추출."""
    if not procedure or pd.isna(procedure):
        return ""
    text = str(procedure)
    dash_lines = re.findall(r"-\s*([^\n\r\-][^\n\r]+)", text)
    if dash_lines:
        return dash_lines[0].strip()[:200]
    return text.strip()[:200]


def is_response_only(utterance: str) -> bool:
    if not utterance:
        return False
    word_count = len(utterance.split())
    if word_count > 8:
        return False
    return bool(RESPONSE_RE.search(utterance))


def has_compound_marker(utterance: str) -> bool:
    if not utterance:
        return False
    return bool(COMPOUND_RE.search(utterance))


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------
def classify_tc(group: pd.DataFrame, title_dup_count: int) -> dict:
    n_steps = len(group)
    first_row = group.iloc[0]
    module_sub_raw = str(first_row.get("Module_Sub") or "")
    module_sub_clean = strip_lang_suffix(module_sub_raw)
    is_multilingual = module_sub_raw != module_sub_clean

    utterances = [extract_utterance(r["Test Procedure"]) for _, r in group.iterrows()]

    compound_module = module_sub_clean in EXPLICIT_COMPOUND_MODULES
    compound_text = any(has_compound_marker(u) for u in utterances)
    is_compound = compound_module or compound_text

    if is_compound:
        pattern = "compound"
        sub_reason = "explicit_module" if compound_module else "connector_in_text"
    elif n_steps == 1:
        pattern = "atomic"
        sub_reason = None
    elif n_steps == 2 and is_response_only(utterances[1]):
        pattern = "confirmation"
        sub_reason = "step2_short_response"
    elif title_dup_count >= 2 and n_steps >= 2:
        pattern = "branching"
        sub_reason = f"title_dup_x{title_dup_count}"
    else:
        pattern = "variant"
        sub_reason = f"steps_{n_steps}"

    return {
        "tc_id": first_row["tc_id"],
        "title": str(first_row["Title"])[:200],
        "module_sub_raw": module_sub_raw,
        "module_sub_clean": module_sub_clean,
        "is_multilingual": is_multilingual,
        "n_steps": n_steps,
        "pattern": pattern,
        "sub_reason": sub_reason,
        "title_dup_count": title_dup_count,
        "utterances": utterances,
    }


def main(input_path: Path, outdir: Path, limit: Optional[int] = None) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"[1/5] Loading {input_path} ...")
    df = pd.read_excel(input_path, sheet_name=0)
    df["tc_id"] = df["_polarion"].ffill()
    df["Title_ffill"] = df["Title"].ffill()
    df["Module_Sub"] = df["Module_Sub"].ffill()

    headers = df[df["Title"].notna()].copy()
    title_counts = headers.groupby("Title").size().to_dict()

    print(f"[2/5] Classifying {headers.shape[0]} TCs ...")
    records = []
    grouped = df.groupby("tc_id", sort=False)
    iter_count = 0
    for tcid, group in grouped:
        title = str(group.iloc[0]["Title"]) if pd.notna(group.iloc[0]["Title"]) else ""
        dup = title_counts.get(title, 1) if title else 1
        rec = classify_tc(group, dup)
        records.append(rec)
        iter_count += 1
        if limit and iter_count >= limit:
            break

    cls_df = pd.DataFrame(records)
    print(f"  → Classified {len(cls_df)} TCs")

    raw_path = outdir / "pattern_classified.parquet"
    try:
        cls_df.to_parquet(raw_path)
    except Exception:
        raw_path = outdir / "pattern_classified.csv"
        cls_df.to_csv(raw_path, index=False)
    print(f"[3/5] Saved raw → {raw_path}")

    total = len(cls_df)
    pattern_counts = cls_df["pattern"].value_counts().to_dict()
    pattern_pct = {k: round(v / total * 100, 2) for k, v in pattern_counts.items()}

    dist = cls_df.pivot_table(
        index="module_sub_clean",
        columns="pattern",
        values="tc_id",
        aggfunc="count",
        fill_value=0,
    )
    dist["total"] = dist.sum(axis=1)
    dist = dist.sort_values(by="total", ascending=False)

    dist_path = outdir / "pattern_distribution.csv"
    dist.to_csv(dist_path)
    print(f"[4/5] Saved module×pattern distribution → {dist_path}")

    library = {
        "_meta": {
            "source": str(input_path),
            "total_tcs": total,
            "pattern_taxonomy": [
                {"name": "atomic", "definition": "단일 step. 일회성 발화 검증."},
                {"name": "variant", "definition": "step ≥ 2, 같은 의도의 평행 발화. step 간 의존성 없음."},
                {"name": "confirmation", "definition": "step1=명령, step2=짧은 응답어 (Yes/No/Run it/취소). 시스템 확인 질문 흐름."},
                {"name": "branching", "definition": "같은 Title을 가진 TC ≥ 2개. step1 동일하고 step2가 분기 응답."},
                {"name": "compound", "definition": "한 step 내에 다중 도메인 발화 (X해주고 Y해줘 / and / 멀티인탠트 module)."},
            ],
            "pattern_counts": pattern_counts,
            "pattern_percent": pattern_pct,
        },
        "patterns": {},
    }

    for pat, sub in cls_df.groupby("pattern"):
        examples = []
        for mod_clean, mod_sub in sub.groupby("module_sub_clean"):
            row = mod_sub.iloc[0]
            examples.append({
                "module_sub": row["module_sub_clean"],
                "tc_id": row["tc_id"],
                "title": row["title"],
                "n_steps": int(row["n_steps"]),
                "utterances": row["utterances"][:5],
                "sub_reason": row["sub_reason"],
            })
            if len(examples) >= 8:
                break

        step_dist = Counter(int(n) for n in sub["n_steps"])
        ml_count = int(sub["is_multilingual"].sum())

        library["patterns"][pat] = {
            "count": int(len(sub)),
            "percent": pattern_pct.get(pat, 0.0),
            "n_steps_distribution": dict(sorted(step_dist.items())),
            "multilingual_count": ml_count,
            "top_modules": sub["module_sub_clean"].value_counts().head(10).to_dict(),
            "examples": examples,
        }

    lib_path = outdir / "pattern_library.json"
    with open(lib_path, "w", encoding="utf-8") as f:
        json.dump(library, f, ensure_ascii=False, indent=2)
    print(f"[5/5] Saved pattern library → {lib_path}")

    print("\n=== Pattern distribution ===")
    for p, c in sorted(pattern_counts.items(), key=lambda x: -x[1]):
        print(f"  {p:<15s} {c:>5d}  ({pattern_pct[p]:5.2f}%)")

    print("\n=== Top modules per pattern ===")
    for p in pattern_counts:
        sub = cls_df[cls_df["pattern"] == p]
        top = sub["module_sub_clean"].value_counts().head(5)
        print(f"\n[{p}]")
        for m, n in top.items():
            print(f"  {m[:50]:<50s} {n}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    main(args.input, args.outdir, args.limit)
