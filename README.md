# Can Learned Firm Weights Beat Market-Cap Weights? A Bottom-up Approach to Next-Day KOSPI Prediction

기업별 수익률 예측을 합쳐 다음 날 KOSPI를 예측할 때, **시장 상태에 따라 학습한 비중**이 **시가총액 비중**보다 나은지를 사전 등록한 검증 절차로 시험한 연구 코드다.

- 저자: Sungwoo Moon, Jeongkyun Lee (Department of Statistics, Korea University)
- 결과 전체: [`docs/RESULTS_KO.md`](docs/RESULTS_KO.md)
- 구현 세부: [`docs/IMPLEMENTATION_KO.md`](docs/IMPLEMENTATION_KO.md)
- 후속 연구: [`followup/`](followup/)

## 핵심 결과

테스트는 2022-01 ~ 2026-10(1,174일)이다. 검증 결과로 모델을 잠근 뒤 한 번만 평가했다.

| 결과 | 값 |
|---|---|
| **연구 질문: 학습 비중(NN-δ) 대 시총 비중(NN-cap)** | MSE −0.04%, Clark–West p = 0.097. **유의하지 않음** |
| 상향식 예측(NN-δ) 대 역사적 평균 | OOS R² 0.44%, p = 0.033. 12개 모델 중 1위이며, 유의하게 이긴 것은 상향식 두 모델뿐 |
| 하락장 뒤 R² (탐색) | NN-δ 1.73%, 외부 기준선 ≤ 0.07% |
| 실패 이유 (진단) | Σpᵢ(rᵢ − R) = 0이라 비중 변경은 사실상 수준 보정(하루 −0.06%p)이 된다. 지수 오차의 99% 이상은 예측할 수 없는 시장 움직임이다 |
| 기업 예측 벤치마크 (KOSPI 2016~2021, 시가 진입 Rank IC) | 이상현상 결합 0.095 > LightGBM 0.070 > 본 모델 0.054 > MASTER 0.042 > DTML 0.030 |

> **The forecasts won; the weights did not.**

## 방법

```
1단계 (기업 예측)  거래 GRU(60일) + 재무 GRU(8분기) → 그래프 어텐션(업종·상관) + 시장 GRU → 다음 날 기업 수익률
                  연도별 교차 적합: k년 예측기는 k년 이전 라벨로만 학습, k년만 예측 (2016~2026)
2단계 (비중)       w = softmax(log p + δ),  손실 = 지수 MSE + τ·KL(w‖p)
                  δ: 선형식 또는 시드별 MLP([기업 벡터, 시장 벡터, 예측 이탈, 시드 간 차이])
                  해마다 중첩 검증으로 τ와 학습 걸음 수 선택 (0걸음 = 시총 비중 허용)
절차              검증 2018~2021 → 잠금(코드·예측 해시 기록) → 테스트 2022~2026 한 번
```

## 폴더 구조

| 경로 | 내용 |
|---|---|
| `code/` | 기본 데이터셋 구축 (`download_marcap.py` → `build_panel.py` → `build_daily_dataset.py`), 공통 데이터 클래스 (`pilot_*.py`) |
| `rev3/code/r3_core.py` | 전처리와 기준선 헤드 함수 (재사용) |
| `rev4/code/` | 수집 `download_*.py`, 전처리 `r4_ingest.py`, 공통 `r4_common.py`, 1단계 `r4_stage1.py`, 2단계 `r4_stage2.py`, 경쟁 모델 `bench_stage1.py`·`bench_eval.py`, KOSDAQ 실행 `run_kosdaq.py` |
| `inputs/` | KOSPI·KOSDAQ 지수 수준, 상장폐지 기준표 |
| `docs/` | 결과 정리, 구현 문서, 전처리·수집 명세, 실험 명세, 실행 순서 |
| `followup/` | 후속 연구 계획과 경쟁 모델 비교 프로토콜 |

**데이터와 결과는 저장소에 없다.** 라이선스와 용량 때문이며, 아래 순서로 다시 만든다.

## 설치

```bash
python -m venv .venv            # Python 3.12
pip install -r requirements.txt
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
```

## 키 설정
- **키 파일:** `NAME=value` 형식의 줄로 쓴다. 이름은 `KRX_AUTH_KEY`, `DART_API_KEY`, `ECOS_API_KEY`, `KRX_ID`, `KRX_PW`다.
- **위치:** 환경변수 `R4_KEYS`로 경로를 주거나, `~/Downloads/key.txt`에 둔다(`rev4/code/keys.py`).
- **보안:** 키 파일은 절대 커밋하지 않는다(`.gitignore`). 값은 로그에 출력되지 않는다.

## 실행 순서

모든 명령은 저장소 루트에서 `python -X utf8 ...`로 실행한다.

```bash
# 0. 기본 시세 데이터셋 (KOSPI)
python code/download_marcap.py --output raw_marcap
python code/build_panel.py --raw raw_marcap --index inputs/market_indices.csv --output panel --daily \
       --first-year 2013 --delisting-reference inputs/delisting_reference.csv
python code/build_daily_dataset.py --daily panel/krx_1995_2026_daily.parquet --raw raw_marcap --no-fundamentals \
       --min-start 2015 --since 2013-01-01 --delisting-reference inputs/delisting_reference.csv --output daily_dataset

# 1. 추가 원천 자료 → rev4/data/raw/
python rev4/code/download_krx.py --vkospi --futures --market --flows   # KRX OPEN API + pykrx 로그인
python rev4/code/download_dart.py --top 300                            # OpenDART (하루 호출 한도 주의)
python rev4/code/download_ecos.py --macro                              # 한국은행 ECOS
python rev4/code/download_us.py                                        # Yahoo Finance
#    업종: KRX KIND 상장법인목록을 rev4/data/raw/krx/sector_YYYYMMDD.csv, kind_listing_YYYYMMDD.csv로 저장

# 2. 전처리 (+ 감사 파일 rev4/data/processed/audit/)
python rev4/code/r4_ingest.py --derived --calendar

# 3. 상관 그래프 캐시 (날짜별 60일 수익률 상관 상위 10개 이웃)
python -c "import sys; sys.path.insert(0,'rev4/code'); from r4_common import *; d,_=load_data4(); CorrelationGraphs.prepare(d, ROOT/'results_daily_2015')"

# 4. 1단계: 교차 적합 기업 예측
python rev4/code/r4_stage1.py --model lin --years 2016-2021
python rev4/code/r4_stage1.py --model nn  --years 2016-2021 --seeds 0,1,2,3,4

# 5. 2단계: 검증 → 잠금
python rev4/code/r4_stage2.py --phase validation
python rev4/code/r4_stage2.py --phase lock

# 6. 테스트 연도 예측과 테스트 (한 번만)
python rev4/code/r4_stage1.py --model lin --years 2022-2026 --allow-test-years
python rev4/code/r4_stage1.py --model nn  --years 2022-2026 --seeds 0,1,2,3,4 --allow-test-years
python rev4/code/r4_stage2.py --phase test
python rev4/code/r4_stage2.py --phase report

# 7. (후속 연구) 기업 예측 경쟁 모델 비교
python rev4/code/bench_stage1.py --models mlp,gru,dtml,master,master_csz,lgb --years 2016-2021 --seeds 0,1,2
python rev4/code/bench_eval.py --market kospi
```

- **소요 시간 (RTX 3050 4GB 기준):** 1단계 신경망은 시드 하나에 약 55분(2016~2026), 2단계는 약 40분이다.
- **KOSDAQ:** 환경변수 `R4_DATA_ROOT`, `R4_RAW`, `R4_PROCESSED`, `R4_GRAPH_DIR`, `R4_FLOWS_FILE`로 경로를 바꿔 같은 코드를 쓴다. 순서는 `rev4/code/run_kosdaq.py`를 참고한다.

## 데이터 출처
- **시세:** FinanceData marcap (고정 커밋)
- **수급·VKOSPI·선물·업종:** KRX (pykrx, KRX OPEN API, KIND)
- **재무:** OpenDART (2015년 이후)
- **금리·환율·거시:** 한국은행 ECOS
- **미국 시장:** Yahoo Finance

각 출처의 이용 약관을 따른다.

## 한계
- 업종 분류는 2026년 스냅숏을 과거에 적용했다.
- 하락장 강점, 외부 기준선 비교, 실패 원인 진단은 잠금 이후의 탐색 결과다.
