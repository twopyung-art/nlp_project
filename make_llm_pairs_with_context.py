import pandas as pd
import re
from pathlib import Path

UTT_LINE_RE = re.compile(r"^\[(?P<lang>[a-z]{2}-[A-Z]{2})\]\s*(?P<utt>.+)$")

def parse_procedure(procedure_text):
    if not isinstance(procedure_text, str):
        return None, None
    ko_utt, en_utt = None, None
    for line in procedure_text.splitlines():
        m = UTT_LINE_RE.match(line.strip())
        if m:
            lang = m.group("lang")
            utt = m.group("utt").strip()
            if lang == "ko-KR": ko_utt = utt
            elif lang == "en-US": en_utt = utt
    return ko_utt, en_utt

def run_prep():
    base_path = Path("output/labse")
    sim_path = base_path / "pair_similarity.parquet"
    raw_path = Path("output/MASSIVE_v2_v1_gemma2_27b_final.xlsx") 

    print("[*] 데이터 로딩 중...")
    df_sim = pd.read_parquet(sim_path)
    df_raw = pd.read_excel(raw_path)

    # --- 핵심 수정 부분: ID 컬럼 타입을 문자열로 통일 ---
    df_sim["a_id"] = df_sim["a_id"].astype(str)
    df_sim["b_id"] = df_sim["b_id"].astype(str)
    df_raw["Original_ID"] = df_raw["Original_ID"].astype(str)
    # ------------------------------------------------

    print("[*] Procedure 파싱 중...")
    parsed_data = df_raw["Procedure"].apply(parse_procedure)
    df_raw["ko-KR"] = parsed_data.apply(lambda x: x[0])
    df_raw["en-US"] = parsed_data.apply(lambda x: x[1])

    print("[*] Gray Zone 필터링 및 다국어 결합 중...")
    gray_pairs = df_sim[(df_sim["similarity"] >= 0.70) & (df_sim["similarity"] < 0.95)].copy()

    context_df = df_raw[['Original_ID', 'ko-KR', 'en-US', 'Expected_Result']]

    # TC_A 결합
    enriched = gray_pairs.merge(context_df, left_on='a_id', right_on='Original_ID')
    enriched = enriched.rename(columns={'ko-KR': 'a_ko', 'en-US': 'a_en', 'Expected_Result': 'a_expected'}).drop(columns=['Original_ID'])

    # TC_B 결합
    enriched = enriched.merge(context_df, left_on='b_id', right_on='Original_ID')
    enriched = enriched.rename(columns={'ko-KR': 'b_ko', 'en-US': 'b_en', 'Expected_Result': 'b_expected'}).drop(columns=['Original_ID'])

    out_path = base_path / "llm_gray_pairs_context.jsonl"
    final_cols = ['intent', 'a_id', 'b_id', 'a_ko', 'a_en', 'a_expected', 'b_ko', 'b_en', 'b_expected', 'similarity']
    enriched[final_cols].to_json(out_path, orient="records", lines=True, force_ascii=False)
    
    print(f"[+] 완료: {len(enriched):,} 쌍 구성")
    print(f"[+] 저장 경로: {out_path}")

def extract_validation_set(input_jsonl, output_excel):
    df = pd.read_json(input_jsonl, lines=True)
    
    # 1. 동작성 도메인(가중치 대상) 판별
    action_prefixes = ('iot', 'alarm', 'timer', 'calendar', 'datetime', 'lists', 'takeaway')
    df['is_action'] = df['intent'].str.lower().str.startswith(action_prefixes)
    df['weight'] = df['is_action'].map({True: 5.0, False: 1.0})
    
    # 2. 유사도 구간별 가중치 (0.85~0.95 사이의 모호한 구간을 더 많이 샘플링)
    df['sim_bin'] = pd.cut(df['similarity'], bins=[0.70, 0.85, 0.95], labels=['Low', 'High'])
    
    # 3. 500개 샘플링
    validation_set = df.sample(n=500, weights='weight', random_state=42)
    
    # 4. 사람이 라벨링하기 편한 엑셀 포맷으로 저장
    # 'is_duplicate' 컬럼을 비워두어 사람이 체크하게 함
    validation_set['is_duplicate(Human)'] = "" 
    validation_set['comment'] = ""
    
    cols = ['intent', 'similarity', 'a_id', 'a_ko', 'a_en', 'b_id', 'b_ko', 'b_en', 'is_duplicate(Human)', 'comment']
    validation_set[cols].to_excel(output_excel, index=False)
    print(f"[+] 검증용 Golden Set(GT) 생성 완료: {output_excel}")
    
if __name__ == "__main__":
    run_prep()
    # extract_validation_set("output/labse/llm_gray_pairs_context.jsonl", "Human_Labeling_Target_500.xlsx")