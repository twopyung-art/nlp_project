import argparse
import pandas as pd
import numpy as np
import umap
import plotly.express as px
from pathlib import Path

def add_guide_text(fig, title_text, low, high):
    """차트 하단에 동적 임계값을 포함한 QE 확인 포인트를 추가합니다."""
    guide_html = (
        f"<br><b>[분석 가이드: {title_text}]</b><br>"
        f"1. <b>설정 임계값:</b> High >= {high:.2f} / Gray {low:.2f}~{high:.2f} / Low < {low:.2f}<br>"
        f"2. <b>분석 포인트:</b> {title_text} 영역의 데이터들이 설정된 기준에 부합하는지 확인하십시오.<br>"
        "3. <b>의사결정:</b> 다른 의도인데 가깝다면 임계값 상향, 같은 의도인데 너무 멀다면 임계값 하향을 검토하십시오."
    )
    fig.add_annotation(
        dict(
            xref="paper", yref="paper", x=0, y=-0.22,
            showarrow=False, text=guide_html,
            align="left", font=dict(size=12)
        )
    )
    fig.update_layout(margin=dict(b=180)) 
    return fig

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--high", type=float, default=0.85)
    ap.add_argument("--low", type=float, default=0.65)
    args = ap.parse_args()

    base_path = Path("output/labse")
    emb_path = base_path / "embeddings.parquet"
    sim_path = base_path / "pair_similarity.parquet"
    dist_path = base_path / "distribution.csv"
    
    df_emb = pd.read_parquet(emb_path)
    df_sim = pd.read_parquet(sim_path)
    df_dist = pd.read_csv(dist_path)
    embed_cols = [c for c in df_emb.columns if c.startswith('e')]

    # --- 1. 전체 데이터 시각화 (동적 가이드 적용) ---
    print("[*] 전체 데이터 시각화 생성 중...")
    reducer_all = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
    embedding_all_2d = reducer_all.fit_transform(df_emb[embed_cols].values)

    df_viz_all = pd.DataFrame({
        "x": embedding_all_2d[:, 0], "y": embedding_all_2d[:, 1],
        "Intent": df_emb["intent"], "KO_Utterance": df_emb["ko_utt"], "ID": df_emb["Original_ID"]
    })
    fig_all = px.scatter(df_viz_all, x="x", y="y", color="Intent", hover_data=["KO_Utterance", "ID"],
                         title=f"All Intents Distribution (High:{args.high}/Low:{args.low})", template="plotly_white", width=1200, height=850)
    fig_all = add_guide_text(fig_all, "전체 데이터 분포", args.low, args.high)
    fig_all.write_html(base_path / "visualize_all_intents.html")

    # --- 2. High 레벨 시각화 (기존 로직) ---
    print(f"[*] High 레벨 분석 중 (Sim >= {args.high})...")
    high_pairs = df_sim[df_sim["bucket"] == "High"]
    high_ids = pd.concat([high_pairs["a_id"], high_pairs["b_id"]]).unique()
    df_high = df_emb[df_emb["Original_ID"].isin(high_ids)].reset_index(drop=True)
    
    if len(df_high) > 1:
        reducer = umap.UMAP(n_neighbors=min(15, len(df_high)-1), min_dist=0.01, metric='cosine', random_state=42)
        embedding_2d = reducer.fit_transform(df_high[embed_cols].values)
        df_viz = pd.DataFrame({"x": embedding_2d[:, 0], "y": embedding_2d[:, 1], "Intent": df_high["intent"], "Function": df_high["function"], "KO_Utterance": df_high["ko_utt"], "ID": df_high["Original_ID"]})
        fig = px.scatter(df_viz, x="x", y="y", color="Intent", symbol="Function", hover_data=["KO_Utterance", "ID"],
                         title=f"High-Level Duplicates (Similarity >= {args.high})", template="plotly_white", width=1200, height=850)
        fig = add_guide_text(fig, "High Zone (중복 후보)", args.low, args.high)
        fig.write_html(base_path / "visualize_high_clusters.html")

    # --- 3. Low 레벨 시각화 (신규 추가) ---
    # Low는 데이터가 너무 많으므로, Intent별로 일부 샘플링하여 경향성 확인
    print(f"[*] Low 레벨 분석 중 (Sim < {args.low})...")
    low_pairs = df_sim[df_sim["bucket"] == "Low"].sample(min(2000, len(df_sim[df_sim["bucket"] == "Low"])))
    low_ids = pd.concat([low_pairs["a_id"], low_pairs["b_id"]]).unique()
    df_low = df_emb[df_emb["Original_ID"].isin(low_ids)].reset_index(drop=True)

    if len(df_low) > 1:
        reducer_low = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
        embedding_low_2d = reducer_low.fit_transform(df_low[embed_cols].values)
        df_viz_low = pd.DataFrame({"x": embedding_low_2d[:, 0], "y": embedding_low_2d[:, 1], "Intent": df_low["intent"], "KO_Utterance": df_low["ko_utt"], "ID": df_low["Original_ID"]})
        fig_low = px.scatter(df_viz_low, x="x", y="y", color="Intent", hover_data=["KO_Utterance", "ID"],
                             title=f"Low-Level Unique Samples (Similarity < {args.low})", template="plotly_white", width=1200, height=850)
        fig_low = add_guide_text(fig_low, "Low Zone (비중복 확인용)", args.low, args.high)
        fig_low.write_html(base_path / "visualize_low_samples.html")

    # --- 4. Focus Domain 상세 분석 (Top 3 Gray + IoT Domain) ---
    # Gray Zone이 많은 일반 Top 3 추출
    general_top3 = df_dist.sort_values("Gray", ascending=False).head(3)["intent"].tolist()
    
    # IoT 관련 인텐트 추출 (iot_ 또는 home_ 키워드 포함)
    iot_intents = [it for it in df_emb["intent"].unique() if "iot_" in it or "home_" in it]
    
    # 분석 대상 결합 (중복 제거)
    focus_intents = list(set(general_top3 + iot_intents))
    print(f"[*] 집중 분석 대상 ({len(focus_intents)}개): {focus_intents}")

    df_focus = df_emb[df_emb["intent"].isin(focus_intents)].reset_index(drop=True)
    
    if len(df_focus) > 1:
        print("[*] Focus Domain UMAP 계산 중...")
        reducer_focus = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
        embedding_focus_2d = reducer_focus.fit_transform(df_focus[embed_cols].values)

        df_viz_focus = pd.DataFrame({
            "x": embedding_focus_2d[:, 0],
            "y": embedding_focus_2d[:, 1],
            "Intent": df_focus["intent"],
            "Function": df_focus["function"],
            "KO_Utterance": df_focus["ko_utt"],
            "ID": df_focus["Original_ID"]
        })

        fig_focus = px.scatter(
            df_viz_focus, x="x", y="y", color="Intent", symbol="Function",
            hover_data=["KO_Utterance", "ID"],
            title=f"Focus Domain Analysis: Top Gray + IoT Intens (High:{args.high})",
            template="plotly_white", width=1300, height=900
        )
        
        # IoT 특화 가이드 추가
        guide_focus = (
            f"<br><b>[IoT/기기제어 분석 가이드]</b><br>"
            "1. <b>슬롯 변이 집중 확인:</b> IoT는 '침실/주방' 같은 장소 슬롯이 핵심입니다. "
            f"유사도 {args.high:.2f} 부근에서 이들이 어떻게 군집화되는지 확인하십시오.<br>"
            "2. <b>기기명 오인식 체크:</b> '조명', '에어컨', '티비' 등 목적어가 다른데 겹쳐 있다면 임계값 상향이 필요합니다.<br>"
            "3. <b>상태 모니터링 vs 제어:</b> '불 켜줘'와 '불 켜져 있어?'가 벡터 공간에서 충분히 분리되어 있는지 점검하십시오."
        )
        fig_focus = add_guide_text(fig_focus, "IoT 및 주요 인텐트 상세 분석", args.low, args.high)
        # 가이드 텍스트 위치 세부 조정
        fig_focus.update_layout(annotations=[dict(text=guide_focus, xref="paper", yref="paper", x=0, y=-0.25, showarrow=False, align="left")])
        
        fig_focus.write_html(base_path / "visualize_focus_domains.html")
        print(f"[+] 집중 분석 시각화 완료: {base_path / 'visualize_focus_domains.html'}")
    
    print(f"\n[+] 모든 시각화 파일 생성 완료: {base_path}")

if __name__ == "__main__":
    main()