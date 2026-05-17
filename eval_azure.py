"""
eval_azure.py — Azure OpenAI로 TC 중복 판별 성능 평가
Human Labeling 500건과 비교해 accuracy / Cohen's κ 산출

Usage:
    python eval_azure.py --samples 100 --label "gpt4o-mini_v2"   # 100건 먼저
    python eval_azure.py --samples 500 --label "gpt4o-mini_v2"   # 전체
    python eval_azure.py --samples 500 --resume                   # 중단 후 재시작
"""

import argparse
import datetime
import json
import re
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
import os
from openai import AzureOpenAI
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix

load_dotenv()

BASE_DIR    = Path(__file__).parent
TC_FILE     = BASE_DIR / "output" / "MASSIVE_v2_v1_gemma2_27b_final.xlsx"
LABELS_FILE = BASE_DIR / "output" / "Human_Labeling_Target_500.xlsx"
HISTORY_FILE = BASE_DIR / "output" / "labse" / "kappa_history.json"
CKPT_FILE   = BASE_DIR / "output" / "azure_eval_checkpoint.jsonl"
RESULT_DIR  = BASE_DIR / "output"

# ── 도메인 정책 (대시보드 v2 동일) ─────────────────────────────────────────────
DOMAIN_POLICIES = {
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
    },
}

BASE_PROMPT = """당신은 AI 스피커 QA 전문가입니다.
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


# ── 헬퍼 ────────────────────────────────────────────────────────────────────────

def extract_short_expected(full_text: str) -> str:
    """Expected_Result에서 언어별 응답 예시 이전까지만 추출 (간결화)"""
    if not isinstance(full_text, str):
        return "N/A"
    cut = re.search(r"\[언어별 응답 예시\]|\*생성형", full_text)
    if cut:
        return full_text[: cut.start()].strip()
    return full_text[:400].strip()


def build_prompt(row: dict) -> str:
    intent = str(row.get("intent", "")).lower()
    domain_key = "general"
    # iot_hue 는 iot 보다 먼저 매칭
    for key in DOMAIN_POLICIES:
        if key in intent:
            domain_key = key
            break

    p = DOMAIN_POLICIES[domain_key]
    dynamic = (
        f"\n### [현재 도메인 특화 정책: {p['name']}]\n"
        f"- 가이드라인: {p['guidelines']}\n"
        f"- 판정 예시: {p['examples']}\n"
    )
    prompt = BASE_PROMPT.replace(
        "### QE 판정 정책 (날씨 및 범용 도메인)", dynamic
    )
    return prompt.format(
        intent=row.get("intent", "N/A"),
        a_id=row.get("a_id", "N/A"),
        a_ko=row.get("a_ko", "N/A"),
        a_en=row.get("a_en", "N/A"),
        a_expected=row.get("a_expected", "N/A"),
        b_id=row.get("b_id", "N/A"),
        b_ko=row.get("b_ko", "N/A"),
        b_en=row.get("b_en", "N/A"),
        b_expected=row.get("b_expected", "N/A"),
    )


def call_azure(client: AzureOpenAI, deployment: str, prompt: str, max_retries: int = 2) -> dict:
    last_err = {"decision": "error", "reasoning": "unknown"}
    for attempt in range(max_retries + 1):
        try:
            t0 = time.time()
            resp = client.chat.completions.create(
                model=deployment,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=200,
            )
            elapsed = round(time.time() - t0, 2)
            text = resp.choices[0].message.content.strip()
            usage = resp.usage

            m = re.search(r"\{.*\}", text, re.DOTALL)
            if m:
                result = json.loads(m.group(0))
                result["_latency"] = elapsed
                result["_prompt_tokens"] = usage.prompt_tokens
                result["_completion_tokens"] = usage.completion_tokens
                return result

            last_err = {
                "decision": "error",
                "reasoning": f"JSON 없음: {text[:80]}",
                "_latency": elapsed,
                "_prompt_tokens": usage.prompt_tokens,
                "_completion_tokens": usage.completion_tokens,
            }
        except json.JSONDecodeError as e:
            last_err = {"decision": "error", "reasoning": f"JSON 파싱 오류: {e}"}
        except Exception as e:
            last_err = {"decision": "error", "reasoning": str(e)}

        if attempt < max_retries:
            time.sleep(1.5)

    return last_err


def load_checkpoint() -> dict:
    processed = {}
    if CKPT_FILE.exists():
        with open(CKPT_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    processed[(obj["a_id"], obj["b_id"])] = obj
    return processed


def append_checkpoint(obj: dict) -> None:
    with open(CKPT_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def load_history() -> list:
    if HISTORY_FILE.exists():
        return json.loads(HISTORY_FILE.read_text("utf-8"))
    return []


def save_history(history: list) -> None:
    HISTORY_FILE.write_text(json.dumps(history, ensure_ascii=False, indent=2), "utf-8")


def kappa_grade(k: float) -> str:
    if k > 0.80: return "거의 완벽함 (Almost Perfect)"
    if k > 0.60: return "높음 (Substantial)"
    if k > 0.40: return "보통 (Moderate)"
    if k > 0.20: return "낮음 (Fair)"
    if k >= 0:   return "매우 낮음 (Slight)"
    return "일치하지 않음 (Poor)"


# ── 메인 ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Azure OpenAI TC 중복 판별 평가")
    parser.add_argument("--samples", type=int, default=100, help="평가 건수 (100 or 500)")
    parser.add_argument("--label", type=str, default=None, help="kappa_history 저장 이름")
    parser.add_argument("--resume", action="store_true", help="체크포인트에서 재시작")
    args = parser.parse_args()

    label = args.label or f"azure-gpt4o-mini_{datetime.date.today()}"

    # Azure 클라이언트 초기화
    endpoint   = os.environ["AZURE_OPENAI_ENDPOINT"]
    api_key    = os.environ["AZURE_OPENAI_API_KEY"]
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
    api_ver    = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
    client = AzureOpenAI(azure_endpoint=endpoint, api_key=api_key, api_version=api_ver)

    print(f"[설정] deployment={deployment}  api_version={api_ver}")
    print(f"[설정] label='{label}'  samples={args.samples}  resume={args.resume}")
    print()

    # ── 데이터 로드 ──────────────────────────────────────────────────────────────
    labels_df = pd.read_excel(LABELS_FILE)
    tc_df     = pd.read_excel(TC_FILE)
    tc_lookup = (
        tc_df.set_index("Original_ID")["Expected_Result"]
        .apply(extract_short_expected)
        .to_dict()
    )

    labels_df["a_expected"] = labels_df["a_id"].map(lambda x: tc_lookup.get(x, "N/A"))
    labels_df["b_expected"] = labels_df["b_id"].map(lambda x: tc_lookup.get(x, "N/A"))
    labels_df["human_label"] = labels_df["is_duplicate(Human)"].apply(
        lambda x: "duplicate" if x is True or x == 1 else "unique"
    )

    sample_df = labels_df.head(args.samples).copy().reset_index(drop=True)

    # ── 체크포인트 ───────────────────────────────────────────────────────────────
    if args.resume:
        processed = load_checkpoint()
        print(f"[재시작] 체크포인트 {len(processed)}건 로드")
    else:
        processed = {}
        if CKPT_FILE.exists():
            CKPT_FILE.unlink()

    results     = list(processed.values())
    skip_keys   = set(processed.keys())
    total_in_tok  = sum(r.get("_prompt_tokens", 0)     for r in results)
    total_out_tok = sum(r.get("_completion_tokens", 0) for r in results)

    remaining = len(sample_df) - len(skip_keys)
    print(f"{'='*65}")
    print(f"평가 시작: 총 {len(sample_df)}건 중 {remaining}건 처리 예정")
    print(f"{'='*65}")

    # ── 추론 루프 ────────────────────────────────────────────────────────────────
    for i, row in sample_df.iterrows():
        key = (row["a_id"], row["b_id"])
        if key in skip_keys:
            print(f"[{i+1:3}/{len(sample_df)}] SKIP")
            continue

        prompt = build_prompt(row.to_dict())
        res    = call_azure(client, deployment, prompt)

        llm_decision = str(res.get("decision", "error")).lower()
        if llm_decision not in ("duplicate", "unique"):
            llm_decision = "error"

        is_correct = (row["human_label"] == llm_decision)

        total_in_tok  += res.get("_prompt_tokens", 0)
        total_out_tok += res.get("_completion_tokens", 0)

        result_row = {
            "intent":       row["intent"],
            "a_id":         row["a_id"],
            "a_ko":         row["a_ko"],
            "b_id":         row["b_id"],
            "b_ko":         row["b_ko"],
            "human_label":  row["human_label"],
            "llm_decision": llm_decision,
            "is_correct":   is_correct,
            "reasoning":    res.get("reasoning", ""),
            "_latency":     res.get("_latency", -1),
            "_prompt_tokens":     res.get("_prompt_tokens", 0),
            "_completion_tokens": res.get("_completion_tokens", 0),
        }
        results.append(result_row)
        append_checkpoint(result_row)

        mark = "✓" if is_correct else "✗"
        a_ko = row["a_ko"][:14].ljust(14)
        b_ko = row["b_ko"][:14].ljust(14)
        print(
            f"[{i+1:3}/{len(sample_df)}] {mark}  {row['intent'][:22]:<22}"
            f"  human={row['human_label']:<10}  llm={llm_decision:<10}"
            f"  A:{a_ko}  B:{b_ko}"
        )

    # ── 결과 집계 ────────────────────────────────────────────────────────────────
    eval_df = pd.DataFrame(results)
    valid   = eval_df[eval_df["llm_decision"].isin(["duplicate", "unique"])].copy()

    print(f"\n{'='*65}")
    print(f"결과 요약  ({len(valid)}건 유효 / {len(eval_df)}건 처리)")
    print(f"{'='*65}")

    if valid.empty:
        print("유효한 판정이 없습니다. 로그를 확인하세요.")
        return

    acc   = accuracy_score(valid["human_label"], valid["llm_decision"])
    kappa = cohen_kappa_score(
        valid["human_label"], valid["llm_decision"],
        labels=["duplicate", "unique"]
    )
    grade = kappa_grade(kappa)

    print(f"  Accuracy     : {acc:.2%}")
    print(f"  Cohen's κ    : {kappa:.4f}  ({grade})")

    cm = confusion_matrix(
        valid["human_label"], valid["llm_decision"],
        labels=["duplicate", "unique"]
    )
    print(f"\n  Confusion Matrix (실제 \\ 예측):")
    print(f"                pred:dup  pred:unique")
    print(f"  actual:dup      {cm[0][0]:4d}       {cm[0][1]:4d}")
    print(f"  actual:unique   {cm[1][0]:4d}       {cm[1][1]:4d}")

    print(f"\n  총 토큰  input={total_in_tok:,}  /  output={total_out_tok:,}")

    # 도메인별 오류율
    print(f"\n  도메인별 오류율 (내림차순):")
    by_intent = (
        eval_df.groupby("intent")["is_correct"]
        .agg(correct="sum", total="count")
        .assign(error_rate=lambda d: 1 - d["correct"] / d["total"])
        .sort_values("error_rate", ascending=False)
    )
    for intent, r in by_intent.iterrows():
        bar = "█" * int(r["error_rate"] * 20)
        wrong = int(r["total"] - r["correct"])
        print(f"  {intent:<30}  {r['error_rate']:4.0%}  ({wrong:2d}/{int(r['total']):2d})  {bar}")

    # kappa_history 저장
    history = load_history()
    history.append({
        "label":       label,
        "model":       f"azure/{deployment}",
        "n_samples":   len(valid),
        "accuracy":    round(acc, 4),
        "kappa":       round(kappa, 4),
        "grade":       grade,
        "timestamp":   datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "prompt_head": BASE_PROMPT[:120].replace("\n", " "),
    })
    save_history(history)
    print(f"\n  ✅ kappa_history.json 저장 완료")

    # 결과 Excel 저장
    out_path = RESULT_DIR / f"azure_eval_{label.replace(' ', '_')}_{len(sample_df)}samples.xlsx"
    with pd.ExcelWriter(out_path, engine="openpyxl") as w:
        eval_df.to_excel(w, sheet_name="전체결과", index=False)
        wrong_df = eval_df[~eval_df["is_correct"]]
        wrong_df.to_excel(w, sheet_name="오류케이스", index=False)
        by_intent.reset_index().to_excel(w, sheet_name="도메인별오류율", index=False)
    print(f"  ✅ 결과 저장: {out_path.name}")


if __name__ == "__main__":
    main()
