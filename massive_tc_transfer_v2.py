"""MASSIVE TC → production-like Korean Smart Speaker TC 변환 (v2).

주요 변경사항 (v1 대비):
  1. Title: [OpenData][Multilingual][AmazonMASSIVE][{scenario}][{intent}] {ko_utt}
     (production-like 4단계 prefix, ~75자)
  2. Function: 21 카테고리 매핑 (intent_text → category 대체)
  3. Expected_Result: production-like 풍부화
     (한글 검증 + 10언어 응답 예시 + 생성형 AI disclaimer)
  4. 3개 프롬프트 버전 (V1/V2/V3) — CLI flag 선택
  5. CLI 인자 (총 처리량 / 모델 / 출력 디렉토리 / resume)
  6. 중간 저장 + resume 지원

사용법:
  # Step 1: 작은 sample (30개) × 3 프롬프트 비교
  python massive_tc_transfer_v2.py --prompt-version v1 --total-count 30
  python massive_tc_transfer_v2.py --prompt-version v2 --total-count 30
  python massive_tc_transfer_v2.py --prompt-version v3 --total-count 30

  # Step 3: 최적 프롬프트 결정 후 전체 처리 (--total-count 0 = ko-KR test 전체)
  python massive_tc_transfer_v2.py --prompt-version v1 --total-count 0 --resume

옵션:
  --prompt-version v1|v2|v3   프롬프트 버전 (기본 v2)
  --total-count N             생성할 TC 수 (0 = 전체, 기본 30)
  --model NAME                Ollama 모델명 (기본 gemma2:27b)
  --output-dir DIR            출력 디렉토리 (기본 output)
  --resume                    기존 xlsx 있으면 이어서 처리
"""

import argparse
import json
import os
import re
import time

import ollama
import pandas as pd
from datasets import load_dataset

# ==========================================
# 1. 설정 — 다국어 + 공통 disclaimer
# ==========================================
LOCALES = [
    "ko-KR", "en-US", "ja-JP", "ar-SA",
    "es-ES", "pt-PT", "fr-FR", "vi-VN", "it-IT", "de-DE",
]

# 모든 Expected_Result 끝에 자동 부착
DISCLAIMER = "*생성형 AI가 생성한 답변의 경우, Test Example 과 다를 수 있음"

# Title prefix (T2 형식)
TITLE_PREFIX = "[OpenData][Multilingual][AmazonMASSIVE]"

# ==========================================
# 2. Function 21 카테고리 매핑 (59 → 21)
# ==========================================
FUNCTION_TO_CATEGORY = {
    # Weather
    "weather_query": "Weather",
    # Calendar
    "calendar_set": "Calendar",
    "calendar_query": "Calendar",
    "calendar_remove": "Calendar",
    # Alarm&Timer
    "alarm_set": "Alarm&Timer",
    "alarm_query": "Alarm&Timer",
    "alarm_remove": "Alarm&Timer",
    # Email
    "email_query": "Email",
    "email_sendemail": "Email",
    "email_querycontact": "Email",
    "email_addcontact": "Email",
    # News
    "news_query": "News",
    # Music
    "play_music": "Music",
    "music_query": "Music",
    "music_likeness": "Music",
    "music_dislikeness": "Music",
    "music_settings": "Music",
    "play_radio": "Music",
    "play_podcasts": "Music",
    "play_audiobook": "Music",
    # Lists
    "lists_query": "Lists",
    "lists_remove": "Lists",
    "lists_createoradd": "Lists",
    # Social
    "social_post": "Social",
    "social_query": "Social",
    # OpenQ&A
    "qa_factoid": "OpenQ&A",
    "qa_definition": "OpenQ&A",
    "qa_currency": "OpenQ&A",
    "qa_maths": "OpenQ&A",
    "qa_stock": "OpenQ&A",
    # DeviceControl
    "iot_hue_lightoff": "DeviceControl(Lighting)",
    "iot_hue_lighton": "DeviceControl(Lighting)",
    "iot_hue_lightup": "DeviceControl(Lighting)",
    "iot_hue_lightdim": "DeviceControl(Lighting)",
    "iot_hue_lightchange": "DeviceControl(Lighting)",
    "iot_coffee": "DeviceControl(Coffee)",
    "iot_cleaning": "DeviceControl(Cleaning)",
    "iot_wemo_on": "DeviceControl(Plug)",
    "iot_wemo_off": "DeviceControl(Plug)",
    # AudioControl
    "audio_volume_up": "AudioControl",
    "audio_volume_down": "AudioControl",
    "audio_volume_mute": "AudioControl",
    "audio_volume_other": "AudioControl",
    # ChitChat
    "general_quirky": "ChitChat",
    "general_joke": "ChitChat",
    "general_greet": "ChitChat",
    # Transport
    "transport_query": "Transport",
    "transport_ticket": "Transport",
    "transport_taxi": "Transport",
    "transport_traffic": "Transport",
    # Cooking
    "cooking_recipe": "Cooking",
    # Takeaway
    "takeaway_query": "Takeaway",
    "takeaway_order": "Takeaway",
    # Recommendation
    "recommendation_events": "Recommendation",
    "recommendation_locations": "Recommendation",
    "recommendation_movies": "Recommendation",
    # DateTime
    "datetime_query": "DateTime",
    "datetime_convert": "DateTime",
    # Game
    "play_game": "Game",
}

# ==========================================
# 3. 3개 프롬프트 버전 (V1/V2/V3)
# ==========================================

DOMAIN_GUIDE_V3 = """
도메인별 검증 가이드 (V3 한정):
- weather: 위치/시간 인식, 날씨 정보(온도/날씨조건/예보) 응답
- calendar: 이벤트 시간/제목 인식, 등록/조회/삭제 동작 + 결과 확인
- alarm: 시간 인식, 알람 등록/제거 + 확인 응답
- play: 곡/채널 선택, 재생 시작 + 곡 정보 응답
- email: 수신함 조회 / 메일 작성 / 연락처 관리
- iot: 장치 상태 변경 (조명/플러그/커피/청소기)
- qa: 질문 의도 분류, 답변 제공 (사실/정의/계산)
- general: 자연스러운 대화 응답
- recommendation: 사용자 맥락 고려한 추천 항목 제공
- news: 뉴스 헤드라인/주제별 뉴스 제공
- music: 곡 정보 / 좋아요 / 재생 제어
- transport: 경로/교통상황/택시/티켓 정보
- audio: 볼륨 변경 동작 확인
- datetime: 현재 시간/날짜/시간대 변환
- lists: 목록 항목 관리 (추가/조회/삭제)
- social: 소셜 미디어 게시/조회
- cooking: 레시피 정보 제공
- takeaway: 음식점 검색/주문
- game: 게임 시작/진행
"""


def _utts_block(item: dict) -> str:
    return "\n".join(f"[{lang}] {item['utts'][lang]}" for lang in LOCALES)


def build_prompt_v1(item: dict) -> str:
    """V1 — Minimal: 의도 인식 + 동작 확인."""
    return f"""당신은 AI 스피커 QE 전문가입니다.
다음 [분석 대상]을 분석하여 테스트 케이스의 [Precondition]과 [Verification]을 한국어로 작성하고,
사용자가 각 언어로 발화 시 시스템 응답 예시 10개를 작성하세요.

[분석 대상]
- 도메인: {item['scenario_text']}
- 의도(intent): {item['intent_text']}
- 다국어 발화 (10개):
{_utts_block(item)}

[작성 가이드 — V1 minimal]
이 테스트는 AI 스피커 End-to-End 검증입니다.
검증 대상: 의도에 맞는 응답 + 동작 확인 (백엔드/응답시간 검증 X).

1. Precondition: 해당 기능 수행에 필요한 장치 상태 (간략, 예: "기기 전원 및 네트워크 연결 상태 확인.")
2. Verification (한글, 검증 항목 2개):
   - '{item['intent_text']}' 의도가 정확히 인식됨
   - 의도에 맞는 동작 수행 및 응답 제공
3. ExampleResponses: 각 언어 발화에 대한 자연스러운 1-2문장 응답 (해당 언어로)

[출력 형식 — JSON 만]
{{
  "Precondition": "...",
  "Verification": "검증 내용 (한글 1-2문장)",
  "ExampleResponses": {{
    "ko-KR": "한국어 응답",
    "en-US": "English response",
    "ja-JP": "...",
    "ar-SA": "...",
    "es-ES": "...",
    "pt-PT": "...",
    "fr-FR": "...",
    "vi-VN": "...",
    "it-IT": "...",
    "de-DE": "..."
  }}
}}"""


def build_prompt_v2(item: dict) -> str:
    """V2 — Standard: 검증 4포인트."""
    return f"""당신은 AI 스피커 QE 전문가입니다.
다음 [분석 대상]을 분석하여 테스트 케이스의 [Precondition]과 [Verification]을 한국어로 작성하고,
사용자가 각 언어로 발화 시 시스템 응답 예시 10개를 작성하세요.

[분석 대상]
- 도메인: {item['scenario_text']}
- 의도(intent): {item['intent_text']}
- 다국어 발화 (10개):
{_utts_block(item)}

[작성 가이드 — V2 standard]
이 테스트는 AI 스피커 End-to-End 검증입니다.
검증 대상: 의도에 맞는 응답 + 동작 확인 (백엔드/응답시간 검증 X).

1. Precondition: 해당 기능 수행에 필요한 장치 상태
2. Verification (한글, 검증 4항목):
   - '{item['intent_text']}' 의도가 정확히 인식됨
   - 발화에서 추출한 정보(시간/장소/대상 등)를 반영한 적절한 동작 수행
   - 사용자에게 자연스러운 음성 응답 제공
   - 다국어 발화 모두에 대해 동일한 의도로 인식되며 해당 언어로 응답
3. ExampleResponses: 각 언어 발화에 대한 자연스러운 1-2문장 응답

[출력 형식 — JSON 만]
{{
  "Precondition": "...",
  "Verification": "검증 내용 (한글, 4개 항목 줄바꿈으로 구분)",
  "ExampleResponses": {{
    "ko-KR": "...",
    "en-US": "...",
    "ja-JP": "...",
    "ar-SA": "...",
    "es-ES": "...",
    "pt-PT": "...",
    "fr-FR": "...",
    "vi-VN": "...",
    "it-IT": "...",
    "de-DE": "..."
  }}
}}"""


def build_prompt_v3(item: dict) -> str:
    """V3 — Domain-aware."""
    return f"""당신은 AI 스피커 QE 전문가입니다.
다음 [분석 대상]을 분석하여 테스트 케이스의 [Precondition]과 [Verification]을 한국어로 작성하고,
사용자가 각 언어로 발화 시 시스템 응답 예시 10개를 작성하세요.

[분석 대상]
- 도메인: {item['scenario_text']}
- 의도(intent): {item['intent_text']}
- 다국어 발화 (10개):
{_utts_block(item)}

[작성 가이드 — V3 domain-aware]
이 테스트는 AI 스피커 End-to-End 검증입니다.
검증 대상: 의도에 맞는 응답 + 동작 확인 (백엔드/응답시간 검증 X).

{DOMAIN_GUIDE_V3}

1. Precondition: 해당 기능 수행에 필요한 장치 상태
2. Verification (한글, '{item['scenario_text']}' 도메인 맞춤 검증 항목):
   - '{item['intent_text']}' 의도가 정확히 인식됨
   - {{도메인 특화 검증 #1: 핵심 동작 — 위 도메인 가이드 참고}}
   - {{도메인 특화 검증 #2: 발화 정보 반영}}
   - {{도메인 특화 검증 #3: 사용자에게 적절한 응답 제공}}
   - 다국어 발화에 대해 해당 언어로 일관된 응답
3. ExampleResponses: 각 언어 발화에 대한 도메인 특화 자연스러운 응답 (1-3 문장)

[출력 형식 — JSON 만]
{{
  "Precondition": "...",
  "Verification": "도메인별 검증 내용 (한글, 5개 항목 줄바꿈으로 구분)",
  "ExampleResponses": {{
    "ko-KR": "...",
    "en-US": "...",
    "ja-JP": "...",
    "ar-SA": "...",
    "es-ES": "...",
    "pt-PT": "...",
    "fr-FR": "...",
    "vi-VN": "...",
    "it-IT": "...",
    "de-DE": "..."
  }}
}}"""


PROMPT_BUILDERS = {
    "v1": build_prompt_v1,
    "v2": build_prompt_v2,
    "v3": build_prompt_v3,
}


# ==========================================
# 4. Expected_Result 조립
#    LLM 응답 (Verification + ExampleResponses) → production-like 텍스트
# ==========================================
def assemble_expected_result(item: dict, llm_data: dict) -> str:
    verification = (llm_data.get("Verification") or "").strip()
    examples = llm_data.get("ExampleResponses") or {}

    parts = [
        f"사용자의 요청에 따라 {item['scenario_text']} 관련 동작이 정상 수행됨을 확인.",
        "",
        "[검증 내용]",
        verification if verification else "(검증 내용 누락)",
        "",
        "[언어별 응답 예시]",
    ]
    for lang in LOCALES:
        resp = (examples.get(lang) or "(응답 예시 누락)").strip()
        parts.append(f"[{lang}] {resp}")
    parts.append("")
    parts.append(DISCLAIMER)

    return "\n".join(parts)


# ==========================================
# 5. TC 생성 (LLM 호출)
# ==========================================
def _parse_llm_json(text: str) -> dict:
    """LLM 응답 JSON robust 파싱.

    LLM(특히 다국어 응답)이 종종 invalid JSON을 뱉음:
      - 문자열 내부 unescaped 줄바꿈 → strict=False 로 허용
      - trailing comma → regex cleanup
      - 일부 누락된 필드 → caller가 .get() 으로 방어
    """
    # 1. 정상 파싱
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 2. strict=False — string 내 control char 허용 (가장 흔한 케이스)
    try:
        return json.loads(text, strict=False)
    except json.JSONDecodeError:
        pass
    # 3. trailing comma 제거 후 재시도
    cleaned = re.sub(r",(\s*[}\]])", r"\1", text)
    try:
        return json.loads(cleaned, strict=False)
    except json.JSONDecodeError:
        pass
    # 4. ```json fenced block 추출 시도
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1), strict=False)
        except json.JSONDecodeError:
            pass
    # 모두 실패
    raise json.JSONDecodeError("All robust parse attempts failed", text, 0)


def generate_tc(item: dict, prompt_version: str, model_name: str,
                max_retries: int = 1) -> tuple:
    prompt = PROMPT_BUILDERS[prompt_version](item)

    last_err = None
    for attempt in range(max_retries + 1):
        try:
            response = ollama.chat(
                model=model_name,
                format='json',
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': 0.1 if attempt == 0 else 0.0},
            )
            llm_data = _parse_llm_json(response['message']['content'])
            break
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            if attempt < max_retries:
                time.sleep(0.5)
                continue
            return None, last_err

    # Function 21 카테고리 대체 (사용자 결정 A1-α)
    category = FUNCTION_TO_CATEGORY.get(item['intent_text'], item['intent_text'])

    # Title (T2 형식: ~75자, production-like 4단계 prefix)
    title = (
        f"{TITLE_PREFIX}"
        f"[{item['scenario_text']}][{item['intent_text']}] "
        f"{item['utts']['ko-KR']}"
    )

    # Procedure: 다국어 통합 (E1-α)
    procedure = (
        "1. 기동어 발화\n"
        "2. 다음 각 언어별 명령어 발화 실행:\n"
        + _utts_block(item)
    )

    # Expected_Result: 한글 검증 + 10언어 응답 예시 + disclaimer
    expected = assemble_expected_result(item, llm_data)

    tc = {
        "Title": title,
        "Precondition": llm_data.get("Precondition", "기기 전원 및 네트워크 연결 상태 확인."),
        "Procedure": procedure,
        "Expected_Result": expected,
        "Module": ", ".join(LOCALES),       # 사용자 결정: 유지
        "Module_Sub": item['intent_text'],   # intent_text raw (snake_case 유지)
        "Function": category,                 # 21 카테고리로 대체
        "Original_ID": str(item['id']),
    }
    return tc, None


# ==========================================
# 6. Main
# ==========================================
def main() -> None:
    parser = argparse.ArgumentParser(description="MASSIVE → MASSIVE_v2 변환 (production-like)")
    parser.add_argument("--prompt-version", choices=["v1", "v2", "v3"], default="v2")
    parser.add_argument("--total-count", type=int, default=30,
                        help="처리할 TC 수 (0 = ko-KR test 전체)")
    parser.add_argument("--model", type=str, default="gemma2:27b",
                        help="Ollama 모델명 (gemma2:27b, gpt-oss:20b 등)")
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--resume", action="store_true",
                        help="기존 xlsx 있으면 이어서 처리")
    args = parser.parse_args()

    json_dir = os.path.join(args.output_dir, args.prompt_version)
    os.makedirs(json_dir, exist_ok=True)

    suffix = f"{args.prompt_version}_{args.model.replace(':', '_').replace('/', '_')}"
    final_xlsx = os.path.join(args.output_dir, f"MASSIVE_v2_{suffix}.xlsx")
    error_path = os.path.join(args.output_dir, f"errors_{suffix}.json")

    # 데이터셋 로드
    print(f"━━ MASSIVE 다국어 데이터셋 로드 ({len(LOCALES)}개국) ━━")
    try:
        ds_ko = load_dataset("AmazonScience/massive", "ko-KR", split='test')
        intent_labels = ds_ko.features["intent"].names
        scenario_labels = ds_ko.features["scenario"].names
        dataset_dict = {
            lang: load_dataset("AmazonScience/massive", lang, split='test')
            for lang in LOCALES
        }
    except Exception as e:
        print(f"❌ 데이터셋 로드 실패: {e}")
        return

    total_avail = len(ds_ko)
    total = total_avail if args.total_count == 0 else min(args.total_count, total_avail)
    print(f"  사용 가능: {total_avail} TCs, 처리 대상: {total} TCs")
    print(f"  프롬프트: {args.prompt_version}, 모델: {args.model}")
    print(f"  출력: {final_xlsx}")
    print()

    # Resume — 기존 xlsx 있으면 done_ids 추출
    done_ids: set = set()
    final_results: list = []
    if args.resume and os.path.exists(final_xlsx):
        try:
            existing = pd.read_excel(final_xlsx)
            done_ids = set(existing['Original_ID'].astype(str))
            final_results = existing.to_dict('records')
            print(f"✓ resume: 기존 {len(done_ids)}개 보존")
        except Exception as e:
            print(f"⚠️  기존 xlsx 읽기 실패: {e}")

    # Existing errors → id 기준 dedup (latest state 유지)
    errors_by_id: dict = {}
    if os.path.exists(error_path):
        try:
            with open(error_path, 'r', encoding='utf-8') as f:
                for e in json.load(f):
                    errors_by_id[str(e["id"])] = e
        except Exception:
            pass
    # 이미 done_ids 에 있는 (= 이전 retry 로 성공) id 는 errors 에서 제거
    for did in list(done_ids):
        errors_by_id.pop(did, None)

    t0 = time.time()
    n_processed = 0
    n_skipped = 0

    for i in range(total):
        item_id = str(ds_ko[i]["id"])
        if item_id in done_ids:
            n_skipped += 1
            continue

        item = {
            "id": ds_ko[i]["id"],
            "intent_text": intent_labels[ds_ko[i]["intent"]],
            "scenario_text": scenario_labels[ds_ko[i]["scenario"]],
            "utts": {lang: dataset_dict[lang][i]["utt"] for lang in LOCALES},
        }

        n_processed += 1
        elapsed = time.time() - t0
        rate = n_processed / elapsed if elapsed > 0 else 0
        remaining = total - i - 1
        eta = remaining / rate if rate > 0 else 0
        print(f"[{i+1}/{total}] ({rate:.2f}/s, ETA {eta:.0f}s) "
              f"{item['intent_text']}: {item['utts']['ko-KR'][:40]}")

        tc, err = generate_tc(item, args.prompt_version, args.model)
        if tc:
            file_name = f"TC_{i+1:05d}_{item['intent_text']}.json"
            with open(os.path.join(json_dir, file_name), "w", encoding="utf-8") as f:
                json.dump(tc, f, indent=2, ensure_ascii=False)
            final_results.append(tc)
            errors_by_id.pop(item_id, None)  # 재시도 성공 → 에러 제거
        else:
            errors_by_id[item_id] = {"id": item_id, "intent": item['intent_text'], "error": err}
            print(f"  ⚠️  실패: {err}")

        # 25개마다 중간 저장
        if (i + 1) % 25 == 0:
            _save_xlsx(final_xlsx, final_results)
            _save_errors(error_path, list(errors_by_id.values()))
            print(f"  ✓ 중간 저장: ok={len(final_results)}, err={len(errors_by_id)}")

        time.sleep(0.1)

    # 최종 저장
    _save_xlsx(final_xlsx, final_results)
    _save_errors(error_path, list(errors_by_id.values()))

    elapsed_total = time.time() - t0
    print()
    print(f"━━ 완료 ━━")
    print(f"  처리: {n_processed} (skipped {n_skipped})")
    print(f"  성공: {len(final_results)}")
    print(f"  실패: {len(errors_by_id)}")
    if elapsed_total > 0 and n_processed > 0:
        print(f"  소요: {elapsed_total:.0f}s ({n_processed/elapsed_total:.2f}/s)")
    print(f"  저장: {final_xlsx}")
    if errors_by_id:
        print(f"  에러 로그: {error_path}")


def _save_xlsx(path: str, rows: list) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    cols = ["Title", "Precondition", "Procedure", "Expected_Result",
            "Module", "Module_Sub", "Function", "Original_ID"]
    df = df[[c for c in cols if c in df.columns]]
    df.to_excel(path, index=False)


def _save_errors(path: str, errors: list) -> None:
    # 빈 list 도 저장 (모두 해결된 경우 → 파일 비움)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(errors, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
