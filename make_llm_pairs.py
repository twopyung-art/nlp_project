import pandas as pd
import numpy as np
from pathlib import Path

def make_llm_pairs():
    # 1. 경로 설정 및 데이터 로드
    base_path = Path("output/labse")
    emb_path = base_path / "embeddings.parquet"
    sim_path = base_path / "pair_similarity.parquet"
    
    if not (emb_path.exists() and sim_path.exists()):
        print(f"[!] 파일을 찾을 수 없습니다: {base_path}")
        return

    print("[*] 데이터 로딩 및 정규화 중...")
    df_emb = pd.read_parquet(emb_path)
    df_sim = pd.read_parquet(sim_path)
    
    # [방어] ID 타입 및 컬럼명 표준화
    df_sim['a_id'] = df_sim['a_id'].astype(str)
    df_sim['b_id'] = df_sim['b_id'].astype(str)
    df_emb['Original_ID'] = df_emb['Original_ID'].astype(str)
    
    # df_emb의 Function 컬럼 찾기 (대소문자 무관)
    func_col = next((c for c in df_emb.columns if c.lower() == 'function'), None)
    if not func_col:
        print("[!] df_emb에서 Function 컬럼을 찾을 수 없습니다.")
        return

    # 2. Gray Zone 필터링 (0.70 <= sim < 0.95)
    gray_pairs = df_sim[(df_sim["similarity"] >= 0.70) & 
                        (df_sim["similarity"] < 0.95)].copy()
    
    if gray_pairs.empty:
        print("[!] Gray Zone에 해당하는 쌍이 없습니다.")
        return

    # 3. 안전한 병합 (Merge) - 충돌 방지를 위해 필요한 컬럼만 추출
    # merge 시 기존에 존재할 수 있는 중복 컬럼 제거
    cols_to_keep = ['a_id', 'b_id', 'similarity']
    gray_pairs = gray_pairs[cols_to_keep]

    # 병합용 메타데이터 준비
    meta = df_emb[["Original_ID", "ko_utt", "intent", func_col]].copy()

    # TC A 정보 결합 및 이름 고정
    gray_pairs = gray_pairs.merge(meta, left_on="a_id", right_on="Original_ID", how="left")
    gray_pairs = gray_pairs.rename(columns={
        "ko_utt": "a_utt", 
        "intent": "a_intent", 
        func_col: "a_func"
    }).drop(columns=["Original_ID"])

    # TC B 정보 결합 및 이름 고정
    gray_pairs = gray_pairs.merge(meta, left_on="b_id", right_on="Original_ID", how="left")
    gray_pairs = gray_pairs.rename(columns={
        "ko_utt": "b_utt", 
        "intent": "b_intent", 
        func_col: "b_func"
    }).drop(columns=["Original_ID"])

    print(f"[*] 병합 완료. 현재 컬럼: {gray_pairs.columns.tolist()}")

    # 4. 동작성 도메인 판별 로직
    def is_action_domain(func_name):
        # 모든 동작성 도메인 접두사 포함
        action_prefixes = ('iot', 'alarm', 'timer', 'calendar', 'datetime', 'lists', 'takeaway')
        return str(func_name).lower().startswith(action_prefixes)

    # 5. 가중치 계산
    print("[*] 동작성 도메인 가중치 계산 중...")
    gray_pairs['is_action'] = gray_pairs.apply(
        lambda x: is_action_domain(x['a_func']) or is_action_domain(x['b_func']), axis=1
    )
    gray_pairs['weight'] = gray_pairs['is_action'].apply(lambda x: 5.0 if x else 1.0)
    
    # 6. 유사도 Binning
    gray_pairs['bin'] = pd.cut(gray_pairs['similarity'], bins=[0.70, 0.80, 0.90, 0.95], labels=['Low', 'Mid', 'High'])

    # 7. Golden Set 추출
    golden_size = 400
    golden_set = gray_pairs.sample(n=min(len(gray_pairs), golden_size), weights='weight', random_state=42)

    # 8. 저장
    out_cols = ["a_intent", "similarity", "a_id", "a_utt", "a_func", "b_id", "b_utt", "b_func"]
    
    # JSONL 저장 (전체 작업용)
    gray_pairs[out_cols].to_json(base_path / "llm_input_pairs.jsonl", orient="records", lines=True, force_ascii=False)
    
    # CSV 저장 (Golden Set 라벨링용)
    golden_set_out = golden_set[out_cols].copy()
    golden_set_out['human_label'] = ""
    golden_set_out['reason'] = ""
    golden_set_out.to_csv(base_path / "golden_set_v6_final.csv", index=False, encoding="utf-8-sig")

    print(f"\n[+] 성공: {len(gray_pairs):,} 쌍 생성됨.")
    print(f"[+] Golden Set: {base_path}/golden_set_v6_final.csv")

if __name__ == "__main__":
    make_llm_pairs()