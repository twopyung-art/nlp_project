import streamlit as st
import pandas as pd
import json
import re
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI
from pathlib import Path
import io
import requests
import os
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix

# --- 1. 페이지 레이아웃 및 환경 설정 ---
if st.runtime.exists():
    st.set_page_config(page_title="QE Evaluation Dashboard", layout="wide")

# --- 2. 데이터 및 설정 경로 ---
BASE_DIR = Path(__file__).parent
# 실제 환경에 맞춰 경로 수정 필요 (예시 경로는 유지)
INPUT_FILE = BASE_DIR / "output" / "labse" / "llm_gray_pairs_context.jsonl"

# --- 3. 도메인 정책 사전 ---
if 'domain_policies' not in st.session_state:
    st.session_state.domain_policies = {
        "weather": {
            "name": "날씨 도메인",
            "guidelines": "지역명만 다른 경우 duplicate, 시간대 다르면 unique",
            "examples": "서울 날씨 vs 용인 날씨 -> duplicate"
        },
        "news": {
            "name": "뉴스 도메인",
            "guidelines": "카테고리 다르면 unique, 표현만 다르면 duplicate",
            "examples": "정치 뉴스 vs 경제 뉴스 -> unique"
        },
        "iot_hue": {
            "name": "조명/IoT 도메인",
            "guidelines": "공간(침실, 거실 등)이 다르면 unique, 대상 기기가 다르면 unique",
            "examples": "침실 조명 꺼 vs 거실 조명 꺼 -> unique"
        },
        "general": {
            "name": "기타/범용",
            "guidelines": "동작/대상 일치 시 duplicate, 어조 차이 무시",
            "examples": "불 켜 vs 조명 켜줘 -> duplicate"
        }
    }

# --- 데이터 로드 로직 ---
@st.cache_data
def load_data():
    if INPUT_FILE.exists():
        try:
            return pd.read_json(INPUT_FILE, lines=True)
        except Exception as e:
            st.error(f"데이터 로드 실패: {e}")
    return pd.DataFrame()

def check_ollama_connection(base_url):
    try:
        url = base_url.replace("/v1", "/api/tags") if "/v1" in base_url else f"{base_url}/api/tags"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            return True, models
        return False, []
    except:
        return False, []

# --- 4. 동적 프롬프트 & LLM 추론 ---
def generate_dynamic_prompt(base_template, intent):
    domain_key = "general"
    if intent:
        intent_lower = intent.lower()
        for key in st.session_state.domain_policies.keys():
            if key in intent_lower:
                domain_key = key
                break
    
    policy = st.session_state.domain_policies[domain_key]
    dynamic_section = f"""
### [현재 도메인 특화 정책: {policy['name']}]
- 가이드라인: {policy['guidelines']}
- 판정 예시: {policy['examples']}
"""
    if "### QE 판정 정책" in base_template:
        return base_template.replace("### QE 판정 정책 (날씨 및 범용 도메인)", dynamic_section)
    return base_template + dynamic_section

def run_llm_inference(client, model_name, prompt_template, row):
    try:
        final_prompt = generate_dynamic_prompt(prompt_template, row.get('intent', ''))
        prompt_content = final_prompt.format(
            intent=row.get('intent', 'N/A'),
            a_id=row.get('a_id', 'N/A'), a_ko=row['a_ko'], a_en=row.get('a_en', 'N/A'), a_history=row.get('a_history', 'N/A'),
            b_id=row.get('b_id', 'N/A'), b_ko=row['b_ko'], b_en=row.get('b_en', 'N/A'), b_history=row.get('b_history', 'N/A')
        )
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt_content}],
            temperature=0.1,
            timeout=120.0
        )
        text = response.choices[0].message.content.strip()
        
        match = re.search(r'(\{.*\})', text, re.DOTALL)
        if match:
            json_str = match.group(1)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as je:
                return {"decision": "error", "reasoning": f"JSON Decode Error: {str(je)}", "raw_response": text}
        
        return {"decision": "error", "reasoning": "No JSON object found in response", "raw_response": text}
        
    except Exception as e:
        return {"decision": "error", "reasoning": f"Inference Error: {str(e)}", "representative_id": "ERROR"}

def get_kappa_interpretation(kappa):
    if kappa < 0: return "일치하지 않음 (Poor)"
    if kappa <= 0.20: return "매우 낮음 (Slight)"
    if kappa <= 0.40: return "낮음 (Fair)"
    if kappa <= 0.60: return "보통 (Moderate)"
    if kappa <= 0.80: return "높음 (Substantial)"
    return "거의 완벽함 (Almost Perfect)"

# --- 5. 메인 UI ---
def main():
    st.title("🧪 QE 검증 및 최적화 대시보드")
    
    df = load_data()

    with st.sidebar:
        st.header("⚙️ 환경 및 모델 설정")
        base_url = st.text_input("Ollama URL", value="http://localhost:11434/v1")
        is_connected, models = check_ollama_connection(base_url)
        
        if is_connected:
            st.success(f"✅ Ollama 연결됨 (모델 {len(models)}개 발견)")
            model_list = [m['name'] for m in models]
            default_idx = 0
            for i, m in enumerate(model_list):
                if "llama3.1" in m:
                    default_idx = i
                    break
            selected_model = st.selectbox("사용할 모델 선택", model_list, index=default_idx)
        else:
            st.error("❌ Ollama 서버에 연결할 수 없습니다.")
            selected_model = "llama3.1:latest"
            
        client = OpenAI(base_url=base_url, api_key="ollama")

        st.divider()
        st.header("🏢 도메인별 가이드 수정")
        target_domain = st.selectbox("편집할 도메인", list(st.session_state.domain_policies.keys()))
        
        st.session_state.domain_policies[target_domain]['guidelines'] = st.text_area(
            f"{target_domain} 가이드라인",
            value=st.session_state.domain_policies[target_domain]['guidelines']
        )
        st.session_state.domain_policies[target_domain]['examples'] = st.text_area(
            f"{target_domain} 예시",
            value=st.session_state.domain_policies[target_domain]['examples']
        )

        st.divider()
        st.header("🛠 프롬프트 에디터")
        active_prompt = st.text_area("Base Prompt Template", value="""당신은 품질 엔지니어입니다. 주어진 두 문장이 기능적으로 동일한지 판별하세요.
응답은 반드시 다른 텍스트 없이 순수하게 아래 구조의 JSON 데이터만 출력해야 합니다.

### QE 판정 정책 (날씨 및 범용 도메인)

[데이터]
- Intent: {intent}
- A: {a_ko}
- B: {b_ko}

응답 JSON 예시:
{{"decision": "duplicate", "reasoning": "내용이 동일함", "representative_id": "선정ID"}}""", height=250)

    tab1, tab2, tab3 = st.tabs(["🧪 단일 테스트", "📈 배치 분석", "🎯 정답셋 비교 검토 (Kappa)"])

    # --- Tab 1: 단일 테스트 ---
    with tab1:
        if not df.empty:
            st.subheader("개별 문장 쌍 테스트")
            idx = st.number_input("데이터 인덱스 선택", 0, len(df)-1, 0)
            row = df.iloc[idx]
            col_a, col_b = st.columns(2)
            col_a.info(f"**A:** {row['a_ko']}")
            col_b.info(f"**B:** {row['b_ko']}")
            
            if st.button("🚀 현재 프롬프트로 실행", key="single_test_btn"):
                with st.spinner("LLM 추론 중..."):
                    res = run_llm_inference(client, selected_model, active_prompt, row)
                    if res.get("decision") == "error":
                        st.error(f"판정 에러: {res.get('reasoning')}")
                    else:
                        st.success("판정 성공")
                        st.json(res)
        else:
            st.warning("분석할 로컬 데이터(jsonl)가 없습니다.")

    # --- Tab 2: 배치 분석 ---
    with tab2:
        st.subheader("📈 전체 데이터 샘플 분석")
        sample_size = st.slider("분석할 샘플 수", 5, 100, 20)
        
        if st.button("📊 배치 분석 시작", key="batch_analysis_btn"):
            if not df.empty:
                sample_df = df.sample(n=min(sample_size, len(df)))
                results = []
                prog = st.progress(0)
                status_text = st.empty()
                
                for i, (_, row) in enumerate(sample_df.iterrows()):
                    status_text.text(f"분석 중 ({i+1}/{sample_size})")
                    res = run_llm_inference(client, selected_model, active_prompt, row)
                    res.update({'a_ko': row['a_ko'], 'b_ko': row['b_ko']})
                    results.append(res)
                    prog.progress((i+1)/sample_size)
                
                res_df = pd.DataFrame(results)
                st.success("배치 분석 완료!")
                st.dataframe(res_df, use_container_width=True)
                if 'decision' in res_df.columns:
                    fig = px.pie(res_df, names='decision', title="판정 결과 분포")
                    st.plotly_chart(fig)
            else:
                st.error("데이터가 로드되지 않았습니다.")

    # --- Tab 3: 정답셋 비교 검토 (Kappa) ---
    with tab3:
        st.subheader("🎯 Human Labeling 데이터 비교 (500건 타겟)")
        st.markdown("엑셀 파일(`Human_Labeling_Target_500.xlsx`)을 업로드하여 Human과 LLM의 일치도를 비교합니다.")
        
        uploaded_file = st.file_uploader("정답셋 엑셀 업로드", type=["xlsx"], key="kappa_uploader")
        
        if uploaded_file:
            gt_raw = pd.read_excel(uploaded_file)
            target_col = 'is_duplicate(Human)'
            
            if target_col in gt_raw.columns:
                gt_df = gt_raw.dropna(subset=[target_col]).copy()
                gt_df['human_label'] = gt_df[target_col].apply(lambda x: 'duplicate' if str(x).upper() == 'TRUE' or x == True else 'unique')
                
                st.success(f"✅ 라벨링된 데이터 {len(gt_df)}건 확인 (Null 제외)")
                
                # 중복 버튼 방지: 탭 내부에만 시작 버튼 배치
                if st.button("🚀 Human vs LLM 비교 분석 시작", key="run_kappa_analysis"):
                    results = []
                    prog = st.progress(0)
                    status_display = st.empty()
                    
                    for i, (_, row) in enumerate(gt_df.iterrows()):
                        status_display.markdown(f"**진행도: {i+1}/{len(gt_df)}** \n- A: `{row['a_ko']}`  \n- B: `{row['b_ko']}`")
                        
                        res = run_llm_inference(client, selected_model, active_prompt, row)
                        
                        row_res = row.to_dict()
                        llm_decision = str(res.get('decision', 'unique')).lower()
                        if llm_decision not in ['duplicate', 'unique']:
                            llm_decision = 'unique'
                        
                        row_res.update({
                            'llm_decision': llm_decision,
                            'llm_reasoning': res.get('reasoning'),
                            'is_correct': (row_res['human_label'] == llm_decision)
                        })
                        results.append(row_res)
                        prog.progress((i+1)/len(gt_df))
                    
                    eval_df = pd.DataFrame(results)
                    status_display.empty()
                    prog.empty()
                    
                    # --- 결과 화면 구성 (이미지 요청사항 반영) ---
                    acc = accuracy_score(eval_df['human_label'], eval_df['llm_decision'])
                    kappa = cohen_kappa_score(eval_df['human_label'], eval_df['llm_decision'], labels=['duplicate', 'unique'])
                    
                    # 1. Metrics 상단 배치
                    m1, m2, m3 = st.columns(3)
                    with m1:
                        st.metric("정확도 (Accuracy)", f"{acc:.2%}")
                    with m2:
                        st.metric("Cohen's Kappa", f"{kappa:.4f}")
                    with m3:
                        st.metric("신뢰도 등급", get_kappa_interpretation(kappa))
                    
                    st.divider()

                    # 2. Confusion Matrix & 불일치 케이스 좌우 배치
                    v1, v2 = st.columns([1, 1.5])
                    with v1:
                        st.markdown("### 📊 Confusion Matrix")
                        cm = confusion_matrix(eval_df['human_label'], eval_df['llm_decision'], labels=['duplicate', 'unique'])
                        fig_cm = go.Figure(data=go.Heatmap(
                            z=cm, 
                            x=['Predict Unique', 'Predict Duplicate'], 
                            y=['Actual Unique', 'Actual Duplicate'],
                            text=cm, 
                            texttemplate="%{text}", 
                            colorscale='Blues',
                            showscale=False
                        ))
                        fig_cm.update_layout(height=400, margin=dict(t=20, b=20, l=20, r=20))
                        st.plotly_chart(fig_cm, use_container_width=True)

                    with v2:
                        st.markdown("### ❌ 불일치 케이스 (False Negatives/Positives)")
                        incorrect = eval_df[eval_df['is_correct'] == False]
                        if not incorrect.empty:
                            # 필요한 컬럼만 추출하여 표시
                            display_cols = ['a_ko', 'b_ko', 'human_label', 'llm_decision']
                            if 'comment' in incorrect.columns: display_cols.append('comment')
                            st.dataframe(incorrect[display_cols], use_container_width=True, height=400)
                        else:
                            st.info("불일치 케이스가 없습니다! (100% 일치)")

                    # 3. 전체 결과 하단 배치
                    st.markdown("### 📝 전체 비교 결과 데이터")
                    st.dataframe(eval_df, use_container_width=True)
                    
                    col_down1, col_down2 = st.columns([1, 5])
                    with col_down1:
                        output = io.BytesIO()
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            eval_df.to_excel(writer, index=False)
                        st.download_button("📂 결과 엑셀 다운로드", output.getvalue(), "eval_kappa_results.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            else:
                st.error(f"엑셀 파일에 `{target_col}` 컬럼이 보이지 않습니다.")

if __name__ == "__main__":
    main()