# MASSIVE 다국어 TC 중복 검출 파이프라인

**NLP 프로젝트 (2026.05)**

Amazon MASSIVE 오픈 데이터를 활용해 다국어 TC(테스트케이스)를 구축하고,  
LaBSE 임베딩 + 로컬 LLM으로 의미론적 중복 TC를 자동 검출하는 파이프라인입니다.

---

## 전체 파이프라인

```
[Stage 1] MASSIVE → TC 변환  (Ollama gemma2:27b)
             massive_tc_transfer_v2.py
                    ↓
       output/MASSIVE_v2_v1_gemma2_27b_final.xlsx  ← 공유 데이터
                    ↓
[Stage 2] LaBSE 다국어 임베딩 + TC 쌍 유사도 산출
             labse_similarity.py  →  labse_visualize.py
                    ↓
       output/labse/pair_similarity.parquet  (High / Gray / Low 분류)
                    ↓
[Stage 2.5] LLM 입력 쌍 구성 (Gray Zone 추출)
             make_llm_pairs_with_context.py
                    ↓
       output/labse/llm_gray_pairs_context.jsonl
                    ↓
[Stage 3] 로컬 LLM 중복 판별 대시보드  (Streamlit)
             llm_gray_pairs_dashboard.py
```

---

## 환경 설정

### 필수 소프트웨어

| 항목 | 버전 | 비고 |
|------|------|------|
| Python | 3.10+ | |
| Ollama | 최신 | https://ollama.com |
| 모델 | `qwen3.6:27b` 또는 `gemma2:27b` 등 | 추론 전 pull 필요 |

```bash
# 1. 가상환경 생성
python3 -m venv venv
source venv/bin/activate

# 2. 의존성 설치
pip install -r requirements.txt

# 3. Ollama 모델 준비 (서버에서)
ollama pull gemma2:27b      # Stage 1 TC 생성용
ollama pull qwen3.6:27b     # Stage 3 중복 판별용 (모델은 변경하며 진행)
```

---

## 공유 데이터 파일

| 파일 | 위치 | 설명 |
|------|------|------|
| `MASSIVE_v2_v1_gemma2_27b_final.xlsx` | `output/` | Stage 1 완료 — 전체 TC (2,794건, 10언어) |
| `Human_Labeling_Target_500.xlsx` | `output/` | Stage 3 Human labeling 대상 500 쌍 |

> **산출물 파일** (`output/labse/`, `output/v1~3/` 등)은 git 제외.  
> 각 Stage 스크립트를 실행하면 로컬에 생성됩니다.

---

## 스크립트 실행 가이드

### Stage 1 — MASSIVE TC 생성

```bash
python massive_tc_transfer_v2.py --prompt-version v1 --total-count 0 --resume
# 출력: output/MASSIVE_v2_v1_gemma2_27b_final.xlsx
```

- HuggingFace `AmazonScience/massive` 데이터셋에서 10개 언어 발화 로드
- Ollama `gemma2:27b`로 각 발화에 대해 Precondition / Verification / ExampleResponses 생성
- TC 포맷: Title / Precondition / Procedure / Expected_Result / Module / Module_Sub / Function / Original_ID
- 프롬프트 버전 v1/v2/v3 비교 후 **v1 확정**
- `--resume` 플래그로 중단 후 이어서 처리 가능

### Stage 2 — LaBSE 유사도 산출

```bash
python labse_similarity.py
# 출력: output/labse/embeddings.parquet, pair_similarity.parquet, distribution.csv

python labse_visualize.py
# 출력: output/labse/visualize_*.html
```

**핵심 파라미터:**

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `--high` | 0.95 | High(중복 후보) 컷오프 |
| `--low` | 0.70 | Low(비중복) 컷오프 |
| `--representative` | mean | 10개 언어 평균 임베딩 |

- `sentence-transformers/LaBSE` 모델 사용 (109개 언어 지원)
- 동일 intent(Module_Sub) 내 TC 쌍만 구성
- 임계값 근거: mean 임베딩 기준 슬롯만 다른 쌍(침실/주방 조명, 빨간/파란색 등)이 명확히 구분되는 값으로 실험 설정

### Stage 2.5 — LLM 입력 쌍 구성

```bash
python make_llm_pairs_with_context.py
# 입력: output/labse/pair_similarity.parquet + MASSIVE_v2_v1_gemma2_27b_final.xlsx
# 출력: output/labse/llm_gray_pairs_context.jsonl  (14,680 Gray 쌍, ko/en 컨텍스트 포함)
```

- Gray 쌍(sim 0.70~0.95)만 추출 → LLM 판별 대상
- 각 쌍에 ko-KR / en-US 발화 컨텍스트 추가

### Stage 3 — 중복 판별 대시보드

```bash
streamlit run llm_gray_pairs_dashboard.py \
  --server.address 0.0.0.0 \
  --server.port 8501 \
  --server.enableXsrfProtection false \
  --server.enableCORS false \
  --server.headless true
```

브라우저 접속: `http://<서버IP>:8501`  
SSH 터널 사용 시: `ssh -N -L 8501:localhost:8501 user@<서버>`

**대시보드 기능:**
- `🧪 Test` 탭: 단건 LLM 판정 — 인덱스 선택 후 프롬프트 즉시 테스트
- `📈 Batch` 탭: 건수 지정 → Ollama 배치 처리 → 체크포인트 자동 저장 → Excel 다운로드
- 도메인별 판정 정책(weather / news / calendar / general) 자동 적용

---

## 진행 이력

| 날짜 | 내용 |
|------|------|
| 2026-04-30 | Stage 1 시작 — 프롬프트 버전 v1/v2/v3 샘플(30건) 비교 |
| 2026-05-02 | 프롬프트 v1 확정, 전체 MASSIVE ko-KR 처리 시작 |
| 2026-05-03 | Stage 1 완료 — `MASSIVE_v2_v1_gemma2_27b_final.xlsx` (2,794 TCs) |
| 2026-05-07 | Stage 2 완료 — LaBSE mean 임베딩, 임계값 High 0.95 / Gray 0.70~0.95 확정 |
| 2026-05-08 | Stage 2.5 완료 — Gray 쌍 14,680건 추출, `llm_gray_pairs_context.jsonl` 생성 |
| 2026-05-08 | `Human_Labeling_Target_500.xlsx` 생성 — 검증용 500쌍 선정 |
| 2026-05-10 | Stage 3 대시보드 개발 시작 — 도메인별 정책, 배치 처리, 체크포인트 구현 |
| 2026-05-12 | git 초기화 및 공동 작업 환경 구성 |
| 2026-05-12 | Human Labeling v1 완료 (500쌍) — Cohen's Kappa 측정 기반 도메인 정책 v2 수립 |
| 2026-05-13 | Stage 3 개선 — LLM 수행시간 표시, retry 로직, Kappa 이력 탭 분리(4탭 구조) |
| 2026-05-13 | Stage 2.5 개선 — `Expected_Result` 컨텍스트 추가, jsonl 재생성 |
| 2026-05-13 | Human Labeling v2 진행 중 — Expected_Result 반영 프롬프트 기반 Kappa 재측정 |

---

## 디렉토리 구조

```
massive_datasets/
├── README.md
├── requirements.txt
├── massive_tc_transfer_v2.py       # Stage 1: TC 생성 (Ollama)
├── labse_similarity.py             # Stage 2: LaBSE 유사도
├── labse_visualize.py              # Stage 2: 시각화
├── make_llm_pairs.py               # Stage 2.5: Gray 쌍 추출 (기본)
├── make_llm_pairs_with_context.py  # Stage 2.5: ko/en 컨텍스트 포함
├── extract_patterns.py             # (참고) TC 패턴 분류 스크립트
└── output/
    ├── MASSIVE_v2_v1_gemma2_27b_final.xlsx  ← git 포함
    └── Human_Labeling_Target_500.xlsx        ← git 포함
    # 아래는 git 제외 — 로컬 실행 후 생성됨
    # ├── labse/     (Stage 2~3 산출물)
    # └── v1~v3/     (Stage 1 TC JSON)
```
