# 4판 실행 순서 (데이터를 받은 뒤)

모든 명령은 프로젝트 폴더(`c:\KOSPI_Daily_2015_Laptop_Rev2`)에서 `.venv\Scripts\python -X utf8 …`로 실행한다.

## 0. 원자료 파일 놓기: `rev4/data/raw/`

열 이름은 아래 표 그대로 맞춘다. 없는 파일은 건너뛰고, 그 묶음 없이 돌아간다.

| 파일 | 열 | 우선 |
|---|---|---|
| `dart/financials.csv` | corp_code, stock_code, bsns_year, reprt_code, fs_div, rcept_no, sj_div, account_id, account_nm, thstrm_amount, thstrm_add_amount (OpenDART 출력 그대로) | ★ |
| `dart/filings.csv` | corp_code, stock_code, report_nm, rcept_no, rcept_dt (정정공시 포함) | ★ |
| `krx/investor_flows_kospi.csv` | Date, Code, ForeignNetValue, InstNetValue, IndivNetValue | ★ |
| `krx/sector_YYYYMMDD.csv` (기준일마다 1개) | AsOfDate, Code, Name, Sector | ★ |
| `krx/market_flows.csv` | Date, ForeignNet, InstNet, IndivNet | ★ |
| `krx/vkospi.csv` | Date, VKOSPI | ★ |
| `ecos/rates.csv` | Date, KTB3Y, KTB10Y, CD91, CorpAAm3Y | ★ |
| `ecos/fx.csv` | Date, USDKRW | ★ |
| `ecos/base_rate.csv` | MeetingDate, BaseRate, Decision (동결 회의 포함) | ★ |
| `global/us_daily.csv` | USDate, SPX, NASDAQ, SOX, VIX | ★ |
| `krx/foreign_ownership.csv` | Date, Code, ForeignShares, ForeignLimitShares, ListedShares | ○ |
| `krx/short_trades.csv` | Date, Code, ShortValue | ○ |
| `krx/short_balance.csv` | BaseDate, Code, ShortBalanceValue | ○ |
| `krx/k200_changes.csv` | AnnounceDate, EffectiveDate, Code, Action(편입·편출) | ○ |
| `krx/market_short.csv` | Date, ShortValue, ShortBalanceValue | ○ |
| `krx/futures_program.csv` | Date, K200Spot, K200Futures, OpenInterest, ForeignFuturesNet, ProgramArbNet, ProgramNonArbNet | ○ |
| `kofia/deposits.csv` | Date, ReleaseDate, CustomerDeposits, MarginLoans | ○ |
| `global/global_extra.csv` | USDate, US10Y, US2Y, WTI, DXY, Copper, Gold | ○ |
| `macro/monthly.csv` | Indicator, Period, ReleaseDate, Value (전년비 등 변환한 값) | ○ |

**변환기에 아직 없는 묶음:**
- ○ 묶음: 기업 이벤트(R18), 잠정실적(R19), 기업집단(R20)
- △ 묶음 전체

자료를 받으면 그때 추가한다.

## 1. 변환과 점검 (몇 분)

```
python -X utf8 rev4/code/r4_ingest.py --derived --calendar
```

- **결과 파일:** `rev4/data/processed/` 아래 다섯 가지가 생긴다.
  - `firm_daily_extra.parquet`
  - `market_daily_extra.parquet`
  - `firm_quarterly.parquet`
  - `relations.parquet`
  - `audit/`
- **관문 점검:** 수집 명세 7절 기준이다.
  - `audit/ingest_report.json`
    - `audit_financial_timing_violations` = 0
    - `audit_us_violations` = 0
    - 계정 매칭 수
  - `audit/firm_feature_coverage_by_year.csv`, `market_feature_coverage_by_year.csv`
    - 2015~2021년 커버리지가 70% 미만인 피처를 기록한다.
- **손 대조:** 시총 상위 10개와 상장폐지 5개 종목의 분기 값과 접수일을 DART 원문과 맞춰 본다.

## 2. 1단계: 기업 예측기, 검증 연도

```
python -X utf8 rev4/code/r4_stage1.py --model lin --years 2016-2021
python -X utf8 rev4/code/r4_stage1.py --model nn  --years 2016-2021 --seeds 0,1,2,3,4
```

- **소요 시간:** 선형은 수 분이다. 신경망은 시드 하나에 약 15~25분, 시드 5개에 약 1.5~2시간이다(RTX 3050 기준 추정).
- **학습 순서:** 시드 단위로 학습하므로 끊겨도 끝난 파일은 건너뛰고 이어서 돈다.

## 3. 2단계: 검증 → 잠금

```
python -X utf8 rev4/code/r4_stage2.py --phase validation
python -X utf8 rev4/code/r4_stage2.py --phase lock
```

- 결과: `rev4/results/stage2/validation/`, `rev4/results/plan_locked.json`
- **잠금 뒤에는 모델, 설정, 피처를 바꾸지 않는다.**

## 4. 테스트 (한 번만)

```
python -X utf8 rev4/code/r4_stage1.py --model lin --years 2022-2026 --allow-test-years
python -X utf8 rev4/code/r4_stage1.py --model nn  --years 2022-2026 --seeds 0,1,2,3,4 --allow-test-years
python -X utf8 rev4/code/r4_stage2.py --phase test
python -X utf8 rev4/code/r4_stage2.py --phase report
```

- **소요 시간:** 신경망은 시드 5개에 약 2~3시간이다(학습 기간이 길어서).
- **결과 파일:** `rev4/results/stage2/test/` 아래
  - `tests.csv`: 연구 질문 검정과 최고 모델 검정
  - `overall.csv`, `by_year.csv`
  - `capture.csv`
  - `weight_diagnostics.csv`

## 5. 포스터 그림

```
python -X utf8 rev4/poster/make_static_figures.py      # Figure 1, 2C·2D·2E, 수식 (결과와 무관, 이미 생성됨)
python -X utf8 rev4/code/r4_figures.py                  # Figure 2A·2B (이번 실행의 검증·입력으로), Figure 3~6
```

- 그림은 `rev4/poster/figures/`에 PDF(벡터)와 PNG(300 dpi)로 저장된다.
- 문구는 `rev4/poster/POSTER_TEXT_EN.md`를 쓴다.
- 결론 문장은 `POSTER_RUN_PLAN.md` 6.4절의 해석 규칙에서 고른다.

## 지켜야 할 것

- 테스트 연도(2022~2026) 예측기는 잠금 파일이 있어야 돈다. 코드가 막는다.
- 테스트 결과를 본 뒤에는 다시 돌리지 않는다. 성능이 나쁘면 나쁜 대로 보고한다.
- 시험 실행은 `--out`을 임시 폴더로 지정해서 `rev4/results/`를 건드리지 않는다.
