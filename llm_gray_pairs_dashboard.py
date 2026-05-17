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
INPUT_FILE = BASE_DIR / "output" / "labse" / "llm_gray_pairs_context.jsonl"
HISTORY_FILE = BASE_DIR / "output" / "labse" / "kappa_history.json"

# --- 3. 도메인 정책 사전 ---
# Human labeling 500건 분석 기반 (2026-05-15 v3 개정)
# 주의: iot_hue 는 iot 보다 먼저 위치해야 매칭 우선순위가 올바름
DEFAULT_DOMAIN_POLICIES = {
    # 주의: iot_hue 는 iot 보다 먼저 위치해야 매칭 우선순위가 올바름
    "weather": {
        "name": "날씨 도메인 (weather_query)",
        "guidelines": (
            "UNIQUE: 두 발화 모두 지역 명시, 서로 다른 지역 (서울 vs 경기도 용인)\n"
            "UNIQUE: 시간 범위 다름 (오늘 vs 내일 vs 주말 vs 이번 주)\n"
            "UNIQUE: 날씨 항목 카테고리 다름 (기온 vs 미세먼지 vs 강수 -- 강수 내 비/눈 차이는 DUPLICATE)\n"
            "DUPLICATE: 현재 위치 vs 위치 미지정 (내 지역 날씨 vs 날씨 어때 -- 같은 의도)\n"
            "DUPLICATE: 두 발화 모두 오늘 날씨 문의 -- 직접/간접 표현 무관 (오늘 우산 필요할까 vs 오늘 스웨터 입을까 -> 둘 다 오늘 날씨 의도)\n"
            "DUPLICATE: 두 발화 모두 특정 지역 날씨 조회 (같은 시간대)"
        ),
        "examples": (
            "서울 날씨 vs 경기도 용인 날씨 -> unique (두 지역 모두 명시, 서로 다름)\n"
            "오늘 날씨 vs 내일 날씨 -> unique (날짜 다름)\n"
            "내일 비가 올까 vs 내일 눈이 오나 -> duplicate (강수 카테고리 동일, 세부 유형만 다름)\n"
            "이번 주 내 지역 날씨 vs 이번 주 날씨 -> duplicate (현재위치 vs 미지정 = 동일 의도)\n"
            "오늘 우산 필요할까 vs 오늘 스웨터 입을까 -> duplicate (둘 다 오늘 날씨 문의)\n"
            "오늘 밖에 추워 vs 오늘 날씨 어때 -> duplicate"
        )
    },
    "iot_hue": {
        "name": "조명 도메인 (iot_hue_*)",
        "guidelines": (
            "UNIQUE: 두 발화 모두 방 이름(침실/거실/주방/베란다 등)을 명시하고 서로 다름\n"
            "UNIQUE: 한 발화만 방 이름 명시, 다른 발화는 공간 미지정 (베란다 불 꺼 vs 조명 꺼)\n"
            "UNIQUE: 단일 동작 vs 다중 동작 결합\n"
            "DUPLICATE: 두 발화 모두 공간 지정 없이 같은 동작 (조명 꺼줘 vs 불 꺼줘)\n"
            "DUPLICATE: '머리 위', '천장' 등 방향/위치 묘사는 방 이름 명시로 보지 않음 (공간 미지정으로 처리)\n"
            "DUPLICATE: lightchange에서 색상만 다름 (빨간 vs 분홍 -- 색상은 파라미터 변형)"
        ),
        "examples": (
            "베란다 불 꺼줘 vs 조명 꺼줄래 -> unique (베란다 방 이름 명시 vs 미지정)\n"
            "침실 조명 꺼 vs 거실 조명 꺼 -> unique (두 방 모두 명시, 서로 다름)\n"
            "조명 꺼줘 vs 불 꺼줘 -> duplicate (둘 다 공간 미지정)\n"
            "조명 낮춰 vs 머리 위 조명 어둡게 해 -> duplicate (방향 묘사는 공간 미지정 처리)\n"
            "빨간 조명 해줘 vs 분홍색으로 바꿔줘 -> duplicate (색상 변형, 같은 lightchange)\n"
            "파란색으로 바꿔줘 vs 파란색 바꾸고 주방 꺼 -> unique (단일 vs 다중 의도)"
        )
    },
    "iot": {
        "name": "IoT 일반 도메인 (iot_cleaning, iot_coffee, iot_wemo)",
        "guidelines": (
            "UNIQUE: 루틴/반복(매일/항상/오전N시에) vs 일회성(지금/오늘)\n"
            "UNIQUE: 디바이스 닉네임 지정 vs 범용 ('몽실이 꺼줘' vs '플러그 꺼줘')\n"
            "UNIQUE: 동작 자체 다름 (켜기 vs 끄기, 작동 vs 초기화)\n"
            "DUPLICATE: 같은 동작, 표현만 다름 (커피 내려줘 vs 커피 한 잔 부탁해)\n"
            "DUPLICATE: 청소기 동작 지시 방식만 다름 (청소기 돌려줘 vs 청소기 작동시켜줘)"
        ),
        "examples": (
            "커피 내려줘 vs 매일 커피 내려줘 -> unique (일회성 vs 루틴)\n"
            "커피 머신이 오전 7시에 커피 만들도록 설정 vs 커피 머신 설정 -> unique (특정 시간 루틴 vs 일반 설정)\n"
            "청소기 돌려줘 vs 로봇 청소기 초기화 -> unique (동작 다름)\n"
            "커피 마시고 싶어 vs 커피 한 잔 부탁해 -> duplicate\n"
            "청소기 돌려줘 vs 청소기 작동시켜줘 -> duplicate"
        )
    },
    "calendar": {
        "name": "일정 도메인 (calendar_set/query/remove)",
        "guidelines": (
            "UNIQUE: 동작 자체 다름 (일정 추가 vs 일정 조회 vs 삭제 vs 리마인더 vs 초대)\n"
            "UNIQUE: 전체 일정 조회 vs 특정 일정 조회/상세 (오늘 일정 뭐있어 vs 오늘 오후 회의 달력에 있어)\n"
            "UNIQUE: 삭제 범위 다름 (모든 일정/특정 기간 삭제 vs 특정 이벤트 하나 삭제)\n"
            "UNIQUE: 조회 범위 다름 (특정 시간대 vs 하루 전체)\n"
            "DUPLICATE: 같은 일정 추가/등록 의도 -- 날짜/사람이 달라도 동일 동작 (수요일 미팅 vs 내일 미팅)\n"
            "DUPLICATE: 같은 리마인더 설정 의도 (하루 전 리마인드 vs 하루 전 알려줘)\n"
            "DUPLICATE: 특정 이벤트 하나 삭제, 표현만 다름"
        ),
        "examples": (
            "오늘 일정 뭐 있어 vs 오늘 오후 회의 달력에 있어 -> unique (전체 조회 vs 특정 일정 확인)\n"
            "모든 일정 삭제 vs 생일 지워줘 -> unique (전체 vs 특정)\n"
            "오늘 오후 두시~네시 일정 vs 하루 동안 일정 -> unique (특정 시간대 vs 하루 전체)\n"
            "수요일 정오 미팅 설정 vs 내일 세시 미팅 설정 -> duplicate (일정 추가 동일)\n"
            "자동차 마감 하루 전 리마인드 vs 생일 하루 전 알려줘 -> duplicate (리마인더 설정)"
        )
    },
    "alarm": {
        "name": "알람 도메인 (alarm_set/query/remove)",
        "guidelines": (
            "UNIQUE: 알람 동작 다름 (설정 vs 조회 vs 삭제 vs 끄기)\n"
            "UNIQUE: 알람 목록 조회 vs 알람 시간 확인 (요청 범주 다름)\n"
            "UNIQUE: 시간이 분단위까지 다름 (6시 vs 6시30분)\n"
            "DUPLICATE: 같은 시간대 알람 설정, 표현만 다름 (6시 알람 설정해 vs 6시에 깨워줘)\n"
            "DUPLICATE: 알람 삭제/취소 표현만 다름 (알람 지워줘 vs 알람 취소)"
        ),
        "examples": (
            "내 알람 뭐 있어 vs 알람 몇 시로 설정했어 -> unique (목록 조회 vs 시간 확인)\n"
            "6시 알람 설정 vs 6시30분 알람 설정 -> unique (시간 다름)\n"
            "6시에 깨워줘 vs 6시 알람 설정해 -> duplicate\n"
            "알람 지워줘 vs 알람 취소 -> duplicate"
        )
    },
    "lists": {
        "name": "목록 도메인 (lists_query/createoradd/remove)",
        "guidelines": (
            "UNIQUE: 목록 내용 확인 vs 목록 개수 확인 (목록 보여줘 vs 목록 몇 개 있어)\n"
            "UNIQUE: 목록 항목 확인 vs 목록 확인 (할일 목록에 뭐 있어 vs 목록 리스트)\n"
            "UNIQUE: 특정 항목 삭제 vs 목록 전체 삭제\n"
            "UNIQUE: 항목 지정 vs 미지정 (이거 삭제 vs 저것 삭제)\n"
            "DUPLICATE: 지정 목록에 항목 추가 -- 목록명/항목명 달라도 동일 동작 (쇼핑 목록에 바나나 vs 잡화 목록에 오렌지)\n"
            "DUPLICATE: 특정 항목 삭제, 표현만 다름\n"
            "DUPLICATE: 목록 확인 요청, 표현만 다름 (목록 보여줘 vs 목록 말해봐)"
        ),
        "examples": (
            "나 목록 몇 개 있어 vs 내가 무슨 목록을 요청했지 -> unique (개수 확인 vs 내용 확인)\n"
            "할일 목록에 뭐 있어 vs 나 리스트 몇 개 있어 -> unique (항목 확인 vs 개수 확인)\n"
            "목록에서 항목 삭제 vs 할일 목록 삭제 -> unique (항목 삭제 vs 목록 삭제)\n"
            "쇼핑 목록에 바나나 추가 vs 잡화 목록에 오렌지 추가 -> duplicate (항목 추가)\n"
            "목록 보여줘 vs 목록 말해봐 -> duplicate"
        )
    },
    "email": {
        "name": "이메일 도메인 (email_query/sendemail)",
        "guidelines": (
            "UNIQUE: 동작 다름 (확인/목록 vs 읽기 vs 발송 vs 연락처 추가)\n"
            "UNIQUE: 이메일 목록 보기 vs 읽기 (목록 vs 내용)\n"
            "UNIQUE: 전체 이메일 vs 특정 발신자 이메일\n"
            "DUPLICATE: 특정 사람 이메일 확인 -- 발신자 이름만 다름\n"
            "DUPLICATE: 특정 사람에게 답장 -- 수신자 이름만 다름"
        ),
        "examples": (
            "최근 이메일 리스트 보여줘 vs 새로 온 이메일 읽어줘 -> unique (목록 vs 읽기)\n"
            "내 새 이메일 확인해 vs 효진이한테 온 메일 있나 -> unique (전체 vs 특정인)\n"
            "진혁에게서 온 이메일 vs 효진이한테 온 이메일 -> duplicate (특정인 이메일 확인)\n"
            "어머니 이메일로 답장 vs 진수 이메일에 답장 -> duplicate (특정인 답장)"
        )
    },
    "play": {
        "name": "미디어 재생 도메인 (play_music/radio/podcasts)",
        "guidelines": (
            "UNIQUE: 플레이리스트 이름이 구체적으로 다름\n"
            "UNIQUE: 특정 노래 vs 장르 노래 (이효리 노래 vs 경쾌한 음악)\n"
            "UNIQUE: 재생 방식 다름 (특정 검색 재생 vs 다음곡 재생)\n"
            "DUPLICATE: 같은 장르 음악 요청, 표현만 다름\n"
            "DUPLICATE: 특정 아티스트/조건 음악 재생, 표현만 다름"
        ),
        "examples": (
            "이효리 노래 라디오에서 재생 vs 경쾌한 음악 틀어 -> unique (특정 노래 vs 장르)\n"
            "내 파티 플레이리스트 vs 한밤중의 사랑 틀어줘 -> unique (플레이리스트 다름)\n"
            "재즈 듣고 싶어 vs 재즈 음악 틀어줘 -> duplicate\n"
            "팟캐스트 다음 쇼 재생 vs 김어준 팟캐스트 틀어줘 -> unique (다음곡 vs 특정 검색)"
        )
    },
    "datetime": {
        "name": "날짜/시간 도메인 (datetime_query/convert)",
        "guidelines": (
            "UNIQUE: 날짜 확인 vs 요일 확인 (현재 날짜 vs 특정 날짜의 요일)\n"
            "UNIQUE: 날짜 조회 vs 시간 조회 (다른 API)\n"
            "DUPLICATE: 특정 지역/도시 현재 시간 조회 -- 도시 달라도 동일 의도 (서울 몇시 vs 시카고 몇시 vs 동부표준시 몇시)\n"
            "DUPLICATE: 현재 날짜/시간, 표현만 다름"
        ),
        "examples": (
            "현재 일월연도 뭐야 vs 올해 6월 27일 무슨 요일이야 -> unique (날짜 확인 vs 요일 확인)\n"
            "서울 지금 몇시 vs 시카고 지금 몇시 -> duplicate (특정 지역 시간, 의도 동일)\n"
            "지금 동부 표준시로 몇시야 vs 몇 시야 -> duplicate (시간 확인 의도 동일)"
        )
    },
    "news": {
        "name": "뉴스 도메인 (news_query)",
        "guidelines": (
            "UNIQUE: 뉴스 카테고리가 명시적으로 다름 (정치 vs 경제)\n"
            "DUPLICATE: 최신 뉴스 검색 의도 -- 뉴스 소스 달라도 동일 (동아 vs 네이버)\n"
            "DUPLICATE: 표현만 다름 (뉴스 보여줘 vs 최신 뉴스 알려줘)"
        ),
        "examples": (
            "정치 뉴스 vs 경제 뉴스 -> unique\n"
            "동아 최신 뉴스 vs 네이버 실검 기사 -> duplicate (최신 뉴스 검색)"
        )
    },
    "general": {
        "name": "기타/범용 도메인",
        "guidelines": (
            "UNIQUE: 최종 동작(Action) 또는 대상(Object)이 다름\n"
            "DUPLICATE: 동작/대상 일치, 어조/표현만 다름 (존댓말/반말 포함)\n"
            "DUPLICATE: 같은 종류 파라미터 변형 (두 화폐 환율, 두 수식, 두 단어 정의 조회)"
        ),
        "examples": (
            "불 켜 vs 조명 켜줘 -> duplicate\n"
            "오목 게임 하자 vs 포커 치자 -> unique (게임 종류 다름)\n"
            "중국 환율 vs 미국 달러 환율 -> duplicate (환율 조회, 화폐만 다름)"
        )
    }
}

if 'domain_policies' not in st.session_state:
    st.session_state.domain_policies = DEFAULT_DOMAIN_POLICIES
if 'policies_version' not in st.session_state:
    st.session_state.policies_version = "v2"

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

def run_llm_inference(client, model_name, prompt_template, row, max_retries=2):
    import time
    final_prompt = generate_dynamic_prompt(prompt_template, row.get('intent', ''))
    prompt_content = final_prompt.format(
        intent=row.get('intent', 'N/A'),
        a_id=row.get('a_id', 'N/A'), a_ko=row['a_ko'], a_en=row.get('a_en', 'N/A'),
        a_expected=row.get('a_expected', 'N/A'), a_history=row.get('a_history', 'N/A'),
        b_id=row.get('b_id', 'N/A'), b_ko=row['b_ko'], b_en=row.get('b_en', 'N/A'),
        b_expected=row.get('b_expected', 'N/A'), b_history=row.get('b_history', 'N/A'),
    )

    last_error = None
    total_elapsed = 0.0

    for attempt in range(1, max_retries + 2):  # 1st try + up to max_retries retries
        try:
            t0 = time.time()
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt_content}],
                temperature=0.1,
                timeout=120.0
            )
            elapsed = round(time.time() - t0, 2)
            total_elapsed = round(total_elapsed + elapsed, 2)
            text = response.choices[0].message.content.strip()

            match = re.search(r'(\{.*\})', text, re.DOTALL)
            if match:
                try:
                    result = json.loads(match.group(1))
                    result["_latency_sec"] = total_elapsed
                    if attempt > 1:
                        result["_retries"] = attempt - 1
                    return result
                except json.JSONDecodeError as je:
                    last_error = {"decision": "error", "reasoning": f"JSON Decode Error: {str(je)}", "raw_response": text}
            else:
                last_error = {"decision": "error", "reasoning": "No JSON object found in response", "raw_response": text}

        except Exception as e:
            last_error = {"decision": "error", "reasoning": f"Inference Error: {str(e)}"}
            total_elapsed = round(total_elapsed + (time.time() - t0 if 't0' in dir() else 0), 2)

    last_error["_latency_sec"] = total_elapsed
    last_error["_retries"] = max_retries
    return last_error

def get_kappa_interpretation(kappa):
    if kappa < 0: return "일치하지 않음 (Poor)"
    if kappa <= 0.20: return "매우 낮음 (Slight)"
    if kappa <= 0.40: return "낮음 (Fair)"
    if kappa <= 0.60: return "보통 (Moderate)"
    if kappa <= 0.80: return "높음 (Substantial)"
    return "거의 완벽함 (Almost Perfect)"

def load_kappa_history() -> list:
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_kappa_history(history: list) -> None:
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

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
                if "exaone" in m:
                    default_idx = i
                    break
            selected_model = st.selectbox("사용할 모델 선택", model_list, index=default_idx)
        else:
            st.error("❌ Ollama 서버에 연결할 수 없습니다.")
            selected_model = "llama3.1:latest"
            
        client = OpenAI(base_url=base_url, api_key="ollama")

        st.divider()
        if st.button("도메인 정책 초기화 (v3 Human labeling 기반)"):
            st.session_state.domain_policies = DEFAULT_DOMAIN_POLICIES
            st.session_state.policies_version = "v3"
            st.rerun()
        st.caption(f"현재 정책 버전: {st.session_state.get('policies_version', 'v1')}")

        st.divider()
        st.header("도메인별 가이드 수정")
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
        default_prompt = """당신은 AI 스피커 QA 전문가입니다.
아래 두 테스트케이스(TC)가 기능적으로 동일한지(duplicate) 또는 다른 시스템 경로를 타는지(unique) 판별하세요.

---
[전체 도메인 공통 기준]

UNIQUE 판정 -- 아래 중 하나라도 해당하면 unique:
1. 동작 자체가 다름 (조회 vs 등록, 삭제 vs 끄기, 추가 vs 확인)
2. 범위가 다름 (전체/모든/다 vs 특정/이/저/하나)
3. 한쪽이라도 특정 공간을 명시하고 두 발화의 공간이 다름 (주방 조명 꺼 vs 조명 꺼줘, 침실 vs 거실)
4. 루틴/반복(매일/항상) vs 일회성(지금/오늘)
5. 다중 의도 결합 ("X 하고 Y 해줘") vs 단일 동작
6. 시간이 분단위까지 구체적으로 다름 (6시 vs 6시30분)
7. 목록/항목 이름이 서로 다르게 지정됨 (새 목록 vs 쇼핑 목록)

DUPLICATE 판정 -- 아래에 해당하면 duplicate:
1. 어휘/표현만 다름 (음소거해줘 vs 무음, 조명 꺼줘 vs 불 꺼줘)
2. 같은 종류 파라미터 변형 (두 도시, 두 사람 이름, 두 색상 -- 동일 API 경로)
3. 존댓말/반말/어조 차이만 있음

---
### QE 판정 정책 (날씨 및 범용 도메인)

---
[입력 데이터]
- Intent: {intent}
- A [{a_id}]
  - 발화(ko): {a_ko}
  - 발화(en): {a_en}
  - Expected: {a_expected}
- B [{b_id}]
  - 발화(ko): {b_ko}
  - 발화(en): {b_en}
  - Expected: {b_expected}

[출력] JSON만 출력, 다른 텍스트 없이:
{{"decision": "duplicate" or "unique", "reasoning": "판정 이유 (한글 1-2문장)", "representative_id": "{a_id} 또는 {b_id}"}}"""

        active_prompt = st.text_area("Base Prompt Template", value=default_prompt, height=350)

    tab1, tab2, tab3, tab4 = st.tabs(["🧪 단일 테스트", "📈 배치 분석", "🎯 정답셋 비교 (Kappa)", "📊 Kappa 이력"])

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
                    latency = res.get("_latency_sec", -1)
                    if res.get("decision") == "error":
                        st.error(f"판정 에러: {res.get('reasoning')}")
                    else:
                        st.success("판정 성공")
                        st.json(res)
                    retries = res.get("_retries", 0)
                    retry_info = f" | 재시도: {retries}회" if retries > 0 else ""
                    st.caption(f"모델: {selected_model} | 수행 시간: {latency}초{retry_info}")
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
        # ── 3-A. 분석 실행 ──────────────────────────────────────────
        st.subheader("Human Labeling 비교 분석")

        c_name, c_file = st.columns([1, 2])
        with c_name:
            run_label = st.text_input("실험 이름", placeholder="예: qwen3.6:35b + v2 프롬프트")
        with c_file:
            uploaded_file = st.file_uploader("정답셋 엑셀 업로드", type=["xlsx"], key="kappa_uploader")

        if uploaded_file:
            gt_raw = pd.read_excel(uploaded_file)
            target_col = 'is_duplicate(Human)'

            if target_col not in gt_raw.columns:
                st.error(f"엑셀 파일에 `{target_col}` 컬럼이 없습니다.")
            else:
                gt_df = gt_raw.dropna(subset=[target_col]).copy()
                gt_df['human_label'] = gt_df[target_col].apply(
                    lambda x: 'duplicate' if str(x).upper() == 'TRUE' or x is True else 'unique'
                )
                st.caption(f"라벨링된 데이터 {len(gt_df)}건")

                if st.button("Human vs LLM 비교 분석 시작", key="run_kappa_analysis"):
                    if not run_label.strip():
                        st.warning("실험 이름을 입력하세요.")
                        st.stop()

                    results = []
                    prog = st.progress(0)
                    status_display = st.empty()

                    for i, (_, row) in enumerate(gt_df.iterrows()):
                        status_display.text(f"진행 중... ({i+1}/{len(gt_df)})  A: {row['a_ko'][:20]}  B: {row['b_ko'][:20]}")
                        res = run_llm_inference(client, selected_model, active_prompt, row)
                        row_res = row.to_dict()
                        llm_decision = str(res.get('decision', 'unique')).lower()
                        if llm_decision not in ['duplicate', 'unique']:
                            llm_decision = 'unique'
                        row_res.update({
                            'llm_decision': llm_decision,
                            'llm_reasoning': res.get('reasoning', ''),
                            'is_correct': (row_res['human_label'] == llm_decision)
                        })
                        results.append(row_res)
                        prog.progress((i + 1) / len(gt_df))

                    status_display.empty()
                    prog.empty()
                    eval_df = pd.DataFrame(results)

                    acc   = accuracy_score(eval_df['human_label'], eval_df['llm_decision'])
                    kappa = cohen_kappa_score(eval_df['human_label'], eval_df['llm_decision'],
                                             labels=['duplicate', 'unique'])

                    # 결과 저장 → 이력
                    import datetime
                    history = load_kappa_history()
                    history.append({
                        "label":       run_label.strip(),
                        "model":       selected_model,
                        "n_samples":   len(gt_df),
                        "accuracy":    round(acc, 4),
                        "kappa":       round(kappa, 4),
                        "grade":       get_kappa_interpretation(kappa),
                        "timestamp":   datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "prompt_head": active_prompt[:120].replace("\n", " "),
                    })
                    save_kappa_history(history)
                    st.session_state["kappa_history"] = history

                    # 결과 표시
                    m1, m2, m3 = st.columns(3)
                    m1.metric("정확도", f"{acc:.2%}")
                    m2.metric("Cohen's Kappa", f"{kappa:.4f}")
                    m3.metric("등급", get_kappa_interpretation(kappa))

                    st.divider()
                    v1, v2 = st.columns([1, 1.5])
                    with v1:
                        st.markdown("**Confusion Matrix**")
                        cm = confusion_matrix(eval_df['human_label'], eval_df['llm_decision'],
                                             labels=['duplicate', 'unique'])
                        fig_cm = go.Figure(data=go.Heatmap(
                            z=cm,
                            x=['Pred: unique', 'Pred: duplicate'],
                            y=['Actual: unique', 'Actual: duplicate'],
                            text=cm, texttemplate="%{text}",
                            colorscale='Blues', showscale=False
                        ))
                        fig_cm.update_layout(height=360, margin=dict(t=10, b=10, l=10, r=10))
                        st.plotly_chart(fig_cm, use_container_width=True)
                    with v2:
                        st.markdown("**불일치 케이스**")
                        incorrect = eval_df[~eval_df['is_correct']]
                        if not incorrect.empty:
                            disp_cols = ['a_ko', 'b_ko', 'human_label', 'llm_decision', 'llm_reasoning']
                            if 'comment' in incorrect.columns:
                                disp_cols.append('comment')
                            st.dataframe(incorrect[disp_cols], use_container_width=True, height=360)
                        else:
                            st.info("불일치 케이스 없음 (100% 일치)")

                    st.markdown("**전체 비교 결과**")
                    st.dataframe(eval_df, use_container_width=True)
                    buf = io.BytesIO()
                    with pd.ExcelWriter(buf, engine='openpyxl') as w:
                        eval_df.to_excel(w, index=False)
                    st.download_button(
                        "결과 엑셀 다운로드",
                        buf.getvalue(),
                        f"kappa_{run_label.strip().replace(' ', '_')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

    # --- Tab 4: Kappa 이력 ---
    with tab4:
        st.subheader("Kappa 이력 비교")

        if "kappa_history" not in st.session_state:
            st.session_state["kappa_history"] = load_kappa_history()

        history = st.session_state["kappa_history"]

        if not history:
            st.info("아직 저장된 이력이 없습니다. Kappa 분석을 실행하면 자동 저장됩니다.")
        else:
            hist_df = pd.DataFrame(history)
            hist_df.index = range(1, len(hist_df) + 1)

            # 비교 차트
            fig_bar = px.bar(
                hist_df.reset_index(),
                x="label", y="kappa", color="model",
                text="kappa", hover_data=["accuracy", "n_samples", "timestamp"],
                labels={"label": "실험", "kappa": "Cohen's Kappa"},
                title="실험별 Cohen's Kappa 비교",
            )
            fig_bar.update_traces(texttemplate="%{text:.3f}", textposition="outside")
            fig_bar.add_hline(y=0.6, line_dash="dot", line_color="green",
                              annotation_text="목표 (0.6)", annotation_position="bottom right")
            fig_bar.update_layout(height=360, showlegend=True)
            st.plotly_chart(fig_bar, use_container_width=True)

            # 이력 테이블
            display_cols = ["label", "model", "n_samples", "kappa", "accuracy", "grade", "timestamp"]
            st.dataframe(hist_df[display_cols], use_container_width=True)

            # 삭제
            st.markdown("**이력 삭제**")
            options = [f"{i}. [{r['timestamp']}] {r['label']} | {r['model']} | κ={r['kappa']}"
                       for i, r in enumerate(history, 1)]
            to_delete = st.multiselect("삭제할 항목 선택", options)
            if st.button("선택 항목 삭제", key="delete_history_btn"):
                del_indices = {int(s.split(".")[0]) - 1 for s in to_delete}
                new_history = [r for i, r in enumerate(history) if i not in del_indices]
                save_kappa_history(new_history)
                st.session_state["kappa_history"] = new_history
                st.rerun()

if __name__ == "__main__":
    main()