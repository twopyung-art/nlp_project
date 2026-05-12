"""Stage 2: LaBSE 임베딩 기반 TC 쌍 유사도 산출.

입력 (필수 컬럼: Title, Procedure, Module_Sub, Function, Original_ID):
  output/MASSIVE_v2_v1_gemma2_27b_final.xlsx

출력 (output/labse/):
  - embeddings.parquet      : TC 별 임베딩 (768d) + 메타
  - pair_similarity.parquet : 동일 intent 내 모든 TC 쌍의 cosine sim + bucket
  - distribution.csv        : intent × bucket 분포

분류 규칙 (제안서 Stage 2):
  - High (중복 후보)      : sim >= --high (기본 0.85)
  - Gray (Stage 3 LLM 대상): --low <= sim < --high  (기본 0.65 ~ 0.85)
  - Low  (비중복)         : sim <  --low  (기본 0.65)

쌍 구성: 동일 intent (Module_Sub) 내 TC × TC. 같은 카테고리(Function)가 아닌 동일 intent
        기준이 PATTERN_LIBRARY 가이드에 정합.

대표 임베딩 옵션:
  --representative ko-KR   : ko-KR 발화 단일 임베딩 (기본, 빠름)
  --representative mean    : 10개 언어 평균 (다국어 정렬 활용)

사용:
  python labse_similarity.py
  python labse_similarity.py --high 0.85 --low 0.65
  python labse_similarity.py --representative mean

의존성: sentence-transformers, torch, pandas, openpyxl, pyarrow
  pip install sentence-transformers pyarrow
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

LOCALES = [
    "ko-KR", "en-US", "ja-JP", "ar-SA",
    "es-ES", "pt-PT", "fr-FR", "vi-VN", "it-IT", "de-DE",
]
DEFAULT_INPUT = Path("output/MASSIVE_v2_v1_gemma2_27b_final.xlsx")
DEFAULT_OUTDIR = Path("output/labse")

UTT_LINE_RE = re.compile(r"^\[(?P<lang>[a-z]{2}-[A-Z]{2})\]\s*(?P<utt>.+)$")


def parse_utterances(procedure: str) -> dict:
    """Procedure 텍스트에서 [lang] utterance 라인 추출."""
    if not isinstance(procedure, str):
        return {}
    out = {}
    for line in procedure.splitlines():
        m = UTT_LINE_RE.match(line.strip())
        if m:
            out[m.group("lang")] = m.group("utt").strip()
    return out


def encode_representative(model, df: pd.DataFrame, representative: str,
                          batch_size: int) -> np.ndarray:
    if representative == "ko-KR":
        sents = df["utts"].apply(lambda d: d.get("ko-KR", "")).tolist()
        return model.encode(
            sents, batch_size=batch_size,
            normalize_embeddings=True, show_progress_bar=True,
            convert_to_numpy=True,
        )

    # mean: 10개 언어 임베딩 평균 후 재정규화
    stacked = []
    for lang in LOCALES:
        sents = df["utts"].apply(lambda d: d.get(lang, "")).tolist()
        e = model.encode(
            sents, batch_size=batch_size,
            normalize_embeddings=True, show_progress_bar=True,
            convert_to_numpy=True,
        )
        stacked.append(e)
    embeds = np.mean(np.stack(stacked), axis=0)
    norms = np.linalg.norm(embeds, axis=1, keepdims=True)
    return embeds / np.where(norms > 0, norms, 1.0)


def pairwise_within_intent(df: pd.DataFrame, embeds: np.ndarray) -> pd.DataFrame:
    rows = []
    for intent, sub in df.groupby("Module_Sub", sort=False):
        idxs = sub.index.to_numpy()
        if len(idxs) < 2:
            continue
        e = embeds[idxs]
        sim = e @ e.T  # normalized → cosine
        n = len(idxs)
        iu = np.triu_indices(n, k=1)
        for i, j in zip(*iu):
            rows.append({
                "intent": intent,
                "function": df.at[idxs[i], "Function"],
                "a_idx": int(idxs[i]),
                "b_idx": int(idxs[j]),
                "a_id": str(df.at[idxs[i], "Original_ID"]),
                "b_id": str(df.at[idxs[j], "Original_ID"]),
                # "a_utt": df.at[idxs[i], "utts"].get("ko-KR", ""),
                # "b_utt": df.at[idxs[j], "utts"].get("ko-KR", ""),
                "a_utt_ko": df.at[idxs[i], "utts"].get("ko-KR", ""),
                "a_utt_en": df.at[idxs[i], "utts"].get("en-US", ""),
                "b_utt_ko": df.at[idxs[j], "utts"].get("ko-KR", ""),
                "b_utt_en": df.at[idxs[j], "utts"].get("en-US", ""),                
                "similarity": float(sim[i, j]),
            })
    return pd.DataFrame(rows)


def bucketize(sim: pd.Series, low: float, high: float) -> pd.Categorical:
    return pd.cut(
        sim,
        bins=[-1.0, low, high, 2.0],
        labels=["Low", "Gray", "High"],
        right=False,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--input", default=str(DEFAULT_INPUT))
    ap.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    ap.add_argument("--representative", choices=["ko-KR", "mean"], default="mean")
    ap.add_argument("--model", default="sentence-transformers/LaBSE")
    ap.add_argument("--high", type=float, default=0.95) #0.85->0.95 ex)침실조명/주방조명, 빨간색/파란색 구분안함
    ap.add_argument("--low", type=float, default=0.70) #0.65->0.75 ex)침실조명/주방조명, 빨간색/파란색 구분안함
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()

    import torch
    from sentence_transformers import SentenceTransformer

    inp = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[load] reading {inp}")
    df = pd.read_excel(inp)
    print(f"[load] {len(df)} rows, columns: {list(df.columns)}")

    df["utts"] = df["Procedure"].apply(parse_utterances)
    miss_ko = df["utts"].apply(lambda d: "ko-KR" not in d).sum()
    if miss_ko:
        print(f"[warn] {miss_ko} rows missing ko-KR utterance, dropping")
        df = df[df["utts"].apply(lambda d: "ko-KR" in d)].reset_index(drop=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[model] {args.model} on {device}")
    model = SentenceTransformer(args.model, device=device)

    print(f"[encode] representative={args.representative}")
    embeds = encode_representative(model, df, args.representative, args.batch_size)
    print(f"[encode] shape={embeds.shape}")

    # embeddings.parquet
    emb_meta = pd.DataFrame({
        "tc_idx": np.arange(len(df), dtype=np.int32),
        "Original_ID": df["Original_ID"].astype(str).values,
        "intent": df["Module_Sub"].values,
        "function": df["Function"].values,
        "ko_utt": df["utts"].apply(lambda d: d.get("ko-KR", "")).values,
    })
    emb_arr = pd.DataFrame(embeds, columns=[f"e{i}" for i in range(embeds.shape[1])])
    emb_out = pd.concat([emb_meta, emb_arr], axis=1)
    emb_out.to_parquet(outdir / "embeddings.parquet", index=False)
    print(f"[save] {outdir / 'embeddings.parquet'}")

    # pair_similarity.parquet
    print("[pairs] computing within-intent pairs ...")
    pdf = pairwise_within_intent(df, embeds)
    pdf["bucket"] = bucketize(pdf["similarity"], args.low, args.high)
    pdf.to_parquet(outdir / "pair_similarity.parquet", index=False)
    print(f"[save] {outdir / 'pair_similarity.parquet'} ({len(pdf):,} pairs)")

    # distribution.csv
    dist = pdf.groupby(["intent", "bucket"], observed=False).size().unstack(fill_value=0)
    for col in ["Low", "Gray", "High"]:
        if col not in dist.columns:
            dist[col] = 0
    dist = dist[["High", "Gray", "Low"]]
    dist["total"] = dist.sum(axis=1)
    dist = dist.sort_values("total", ascending=False)
    dist.to_csv(outdir / "distribution.csv")
    print(f"[save] {outdir / 'distribution.csv'}")

    print()
    print("=== bucket 분포 (전체 쌍) ===")
    counts = pdf["bucket"].value_counts().reindex(["High", "Gray", "Low"], fill_value=0)
    total = int(counts.sum())
    for b, n in counts.items():
        pct = (100.0 * n / total) if total else 0.0
        print(f"  {b:5s}: {n:>8,}  ({pct:5.2f}%)")
    print(f"  total: {total:>8,}")
    print()
    print(f"[next] Stage 3 LLM 대상: bucket == 'Gray' (sim {args.low}~{args.high})")
    print(f"        → {outdir / 'pair_similarity.parquet'} 의 Gray 쌍을 입력으로 사용")


if __name__ == "__main__":
    main()
