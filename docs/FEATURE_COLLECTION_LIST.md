# 4판 데이터셋 수집 명세 — 처음부터 다시 모으기

작성일 2026-11-20. 이 문서만 보고 데이터셋 전체를 원천부터 다시 만들 수 있게 쓴다. 기존 `daily_dataset/`의 정의(2판 명세 2~3절, `code/build_panel.py`, `code/build_daily_dataset.py`, `code/build_daily_targets.py`)를 그대로 옮기고, 4판에서 새로 넣는 자료를 더했다.

**표기**
- 우선순위
  - ★ 필수: 없으면 4판 구조(재무 GRU, 거래 GRU, 업종 GNN, 시장 GRU)나 기존 기준선 재현이 성립하지 않는다.
  - ○ 권장: 선행연구 근거가 있고 수집이 쉽다. 2단계 선형 검사에 넣는다.
  - △ 선택: 수집 비용이 크거나 근거가 약하다. 2단계에서 ★·○가 효과를 보이면 그때 모은다.
- 수집 주체
  - **사용자**: 계정, API 키, 로그인이 필요해서 직접 받는다.
  - **자동**: 공개 자료이거나 다른 자료로 계산할 수 있어서 Claude가 받거나 만든다.

---

## 1. 범위

| 항목 | 값 | 이유 |
|---|---|---|
| 대상 | 코스피 보통주, **상장폐지 종목 포함** | 생존편향 방지 |
| 제외 | 코스닥, 코넥스, 우선주, 스팩, 리츠, 인프라펀드(맥쿼리인프라, KB발해인프라), 선박투자회사, 이름이 `…신`으로 끝나는 신주인수권 종목 | 기존 `build_panel.py`의 보통주 판정과 같음 |
| 일별 자료 기간 | **2013-01-02 ~ 2026-10-30** | 표본 첫날 2015-01-02에 252거래일 피처와 60거래일 창이 모두 차려면 약 312거래일 앞이 필요 |
| 재무 자료 기간 | **2011 사업연도 1분기 ~ 2026년 3분기 보고서**(2026-10-30까지 접수된 것) | 2015년 초에 8분기 창, TTM, 전년 동기 비교가 모두 차야 함 |
| 예측 시점 | t일 15:30 장 마감 뒤. 타깃은 t → t+1 수익률 | |

---

## 2. 공통 규칙 (모든 자료)

### 2.1 시점 규칙: t일 피처로 쓸 수 있는 값

| 자료 | t일에 쓰는 값 |
|---|---|
| 국내 일별 시세·수급·VKOSPI·금리·환율 | t일 값 (장 마감 시점에 확정) |
| 미국 시장 | 미국 날짜가 한국 날짜 t보다 **엄격히 앞선** 거래일 중 가장 최근 종가. 미국 t일 장은 한국 t일 장 마감 뒤에 열린다 |
| 아시아 시장(니케이·상하이·항셍) | t−1일 종가. 장 마감 시각이 코스피와 같거나 늦다 |
| 정기보고서 재무 | **최초 접수일(정정 전 원공시)의 다음 거래일부터** |
| 잠정실적 공정공시 | 공시일 다음 거래일부터 |
| 공매도 잔고 | 기준일에 공시 시차(보통 2거래일)를 더한 날부터. 시차는 수집할 때 확인해 기록한다 |
| 고객예탁금·신용융자(시장 전체) | 발표일 다음 거래일부터(보통 1~2영업일 늦음) |
| 월별 거시 지표 | 발표일 다음 거래일부터. 발표일을 모르면 4.3절 표의 보수적 날짜 |
| 기준금리 | 금통위 결정일 당일부터(오전 발표) |
| 업종·기업집단 | 기준일(지정일) 다음 거래일부터 다음 시점 자료가 나올 때까지 |

발표·공시 시각을 모르면 **다음 거래일부터** 쓴다.

### 2.2 저장 규칙

1. **발표일을 같이 저장한다.** 재무는 접수번호(`rcept_no`)와 접수일(`rcept_dt`)을, 거시 지표는 발표일을 저장한다. `rcept_no`의 앞 8자리는 접수일(YYYYMMDD)이다.
2. **종목코드는 6자리 문자열로 저장한다.** 앞자리 0을 지우지 않는다(예: `005930`).
3. **결측은 빈칸으로 둔다.** 0으로 채우지 않는다.
4. **단위**
   - 금액: 원
   - 금리: %
   - 지수: 원래 수준
   - 수익률: 소수 (0.01 = 1%)
5. **수정 이력:** 최초 발표치를 쓴다. 최종 수정치만 받을 수 있으면 그 사실을 `manifest`에 기록한다.
6. **형식**
   - 기업 자료: `Date, Code, …` 롱 포맷
   - 시장 자료: `Date, …`
   - CSV(UTF-8) 또는 Parquet
7. **수집 기록:** 자료마다 `raw/manifest.csv`에 다음을 한 줄씩 남긴다.
   - 출처, 받은 날짜, 조회 조건(기간, 시장, 옵션), 행 수, 비고
8. **규제 구간:** 아래 날짜를 그대로 기록하고 시장 더미로 만든다(4.2절).
   - 공매도 전면 금지: 2020-03-16 ~ 2021-05-02, 2023-11-06 ~ 2025-03-30
   - 부분 재개: 2021-05-03 ~ 2023-11-05에는 KOSPI200·KOSDAQ150 종목만 허용
   - 가격제한폭: 2015-06-15부터 ±30%(그 전 ±15%)

---

## 3. 원천 자료 목록: 무엇을 받아 오나

모든 파일은 `rev4/data/raw/` 아래에 둔다. 변환기가 읽는 정확한 파일 이름과 열은 `RUNBOOK.md` 0절 표가 기준이다(○ 묶음 파일 이름 포함).

### 3.1 ★ 필수

| ID | 자료 | 주체 | 출처 | 저장 파일 | 열 |
|---|---|---|---|---|---|
| R1 | 일별 시세 | 자동 | FinanceData/marcap 연도별 파일(`code/download_marcap.py`, 키 불필요) | `raw/marcap/marcap-YYYY.parquet` | Date, Code, Name, Market, Dept, Open, High, Low, Close, Changes, ChangesRatio, Volume, Amount, Marcap, Stocks |
| R2 | KOSPI 지수 종가 | 자동(보유) | 기존 `inputs/market_indices.csv` | 그대로 | date, kospi, kosdaq |
| R3 | 상장폐지 기준표 | 자동(보유) | 기존 `inputs/delisting_reference.csv`(FinanceDataReader KRX-DELISTING) | 그대로 | Symbol, Name, Market, SecuGroup, Kind, ListingDate, DelistingDate, Reason, Industry |
| R4 | 정기보고서 재무제표 | 사용자 | OpenDART 단일회사 전체 재무제표 API. 연결(CFS)을 먼저 받고, 없을 때만 별도(OFS)를 받는다 | `raw/dart/financials.csv` | corp_code, stock_code, bsns_year, reprt_code, fs_div, rcept_no, sj_div, account_id, account_nm, thstrm_amount, thstrm_add_amount, currency. **API가 주는 열을 가공하지 말고 그대로 저장** |
| R5 | 정기공시 목록 | 사용자 | OpenDART 공시검색 API(정기공시) | `raw/dart/filings.csv` | corp_code, stock_code, report_nm, rcept_no, rcept_dt. 정정공시(`[기재정정]`)도 모두 받는다. 최초 접수일을 여기서 찾는다 |
| R6 | 종목별 투자자 순매수 | 사용자 | KRX(pykrx, 기존 `code/download_investor_flows.py`). KRX 로그인 필요 | `raw/krx/investor_flows_kospi.csv` | Date, Code, ForeignNetValue, InstNetValue, IndivNetValue(+ 각 Buy·SellValue). 투자자: 외국인, 기관합계, 개인 |
| R7 | 업종 분류 | 사용자 | KRX(pykrx `get_market_sector_classifications`) | `raw/krx/sector_YYYYMMDD.csv` | AsOfDate, Code, Name, Sector. 기준일은 2013~2025년 각 해의 마지막 거래일과 2026-10-30(14개 시점) |
| R8 | 시장 전체 투자자 순매수 | 사용자 | KRX(pykrx `get_market_trading_value_by_date`, 시장 KOSPI) | `raw/krx/market_flows.csv` | Date, ForeignNet, InstNet, IndivNet (순매수대금, 원) |
| R9 | VKOSPI | 사용자 | KRX 지수(코스피 200 변동성지수) | `raw/krx/vkospi.csv` | Date, VKOSPI (종가) |
| R10 | 시장금리(일별) | 사용자 | 한국은행 ECOS "시장금리(일별)" 통계표 | `raw/ecos/rates.csv` | Date, KTB3Y(국고채 3년), KTB10Y(국고채 10년), CD91(CD 91일), CorpAAm3Y(회사채 3년 AA−) |
| R11 | 원/달러 환율 | 사용자 | ECOS 환율(일별). 15:30 서울 외환시장 종가가 있으면 그것을 쓴다 | `raw/ecos/fx.csv` | Date, USDKRW, (종류: 종가 또는 매매기준율) |
| R12 | 기준금리·금통위 | 사용자 | ECOS 한국은행 기준금리, 한국은행 통화정책방향 결정 목록 | `raw/ecos/base_rate.csv` | MeetingDate, BaseRate(결정 후), Decision(인상·동결·인하). **동결한 회의도 모두 포함** |
| R13 | 미국 지수 | 자동(키 불필요. 실패하면 사용자) | Yahoo: ^GSPC(S&P 500), ^IXIC(나스닥 종합), ^SOX(필라델피아 반도체), ^VIX. FRED의 SP500은 최근 10년치만 있어 쓰지 않는다 | `raw/global/us_daily.csv` | USDate, SPX, NASDAQ, SOX, VIX (종가) |

### 3.2 ○ 권장

| ID | 자료 | 주체 | 출처 | 열 |
|---|---|---|---|---|
| R14 | 외국인 보유 | 사용자 | KRX(외국인 보유량·한도소진률) | Date, Code, ForeignShares, ForeignLimitShares, ListedShares |
| R15 | 종목별 공매도 | 사용자 | KRX(공매도 거래, 공매도 잔고. 잔고 공시는 2016-06부터) | 거래: Date, Code, ShortValue, ShortVolume. 잔고: BaseDate, Code, ShortBalanceShares, ShortBalanceValue |
| R16 | 시장 공매도 | 사용자 | KRX | Date, ShortValue, ShortBalanceValue |
| R17 | KOSPI200 구성 변경 | 사용자 | KRX 정기·수시 변경 공지 | AnnounceDate, EffectiveDate, Code, Action(편입·편출) |
| R18 | 기업 이벤트 | 사용자 | KRX·DART | Date, Code, EventType(배당락, 권리락, 유상증자, 무상증자, 액면분할, 관리종목 지정·해제, 투자경고·위험, 거래정지·재개) |
| R19 | 잠정실적 공정공시 | 사용자 | OpenDART 공시검색(거래소공시 중 "영업(잠정)실적") | rcept_no, rcept_dt, stock_code, 대상 분기, 매출, 영업이익, 순이익 |
| R20 | 기업집단 | 사용자 | 공정거래위원회 기업집단포털(매년 5월 지정) | Year, DesignationDate, GroupName, CompanyName, Code |
| R21 | 선물·프로그램 매매 | 사용자 | KRX | Date, K200Spot, K200Futures(최근월물 종가), OpenInterest, ForeignFuturesNet, ProgramArbNet, ProgramNonArbNet |
| R22 | 고객예탁금·신용융자(시장 전체) | 사용자 | 금융투자협회 FreeSIS | Date, ReleaseDate, CustomerDeposits, MarginLoans |
| R23 | 해외 금리·달러·원자재 | 자동 | FRED: DGS10, DGS2, DCOILWTICO. Yahoo: DX-Y.NYB(달러인덱스), HG=F(구리), GC=F(금) | USDate, US10Y, US2Y, WTI, DXY, Copper, Gold |
| R24 | 월별 거시 | 사용자 | ECOS, KOSIS, 관세청·산업통상자원부 | Indicator, Period(대상 월), ReleaseDate, Value. 지표: 소비자물가(전년비, 근원), 수출(전년비), 20일 잠정수출, 실업률, 취업자 증감, 산업생산, 경기선행지수 순환변동치, 소비자심리지수(CCSI), 기업경기실사지수(BSI) |

### 3.3 △ 선택

| ID | 자료 | 출처 |
|---|---|---|
| R25 | 기관 세부 6분류(금융투자, 연기금, 투신, 사모, 보험, 은행) 종목별 순매수 | KRX |
| R26 | 종목별 신용융자 잔고 | 금융투자협회·KRX |
| R27 | WICS·GICS 중분류 | FnGuide 등(유료) |
| R28 | 공급망: 사업보고서의 주요 고객·공급사 | DART 사업보고서 본문 |
| R29 | 원/엔, 원/위안 | ECOS |
| R30 | 니케이 225, 상하이 종합, 항셍 | Yahoo(자동) |
| R31 | 미국 FOMC 결정일, 미국 CPI·고용 발표일 | 연준, BLS |

---

## 4. 피처 정의

모든 피처는 2.1절 시점 규칙을 따른다. 정의에 쓰는 기호는 다음과 같다.
- **수정종가:** 기준가 비율로 이은 가격이다.
  - 그날 수익률은 Close ÷ (Close − Changes) − 1이다.
  - 이 값이 ChangesRatio와 0.021%p 넘게 어긋나면 가격 연결을 끊고 새 구간을 시작한다.
  - 롤링 계산은 같은 구간 안에서만 한다.
- **시총:** 그 종목의 Marcap이다.
- **기업 시총:** 같은 기업의 모든 상장 주식(보통주와 우선주) 시총 합이다. 종목코드 앞 5자리가 같은 종목을 더한다.

### 4.1 거래 GRU 입력 (기업 × 일, 최근 60거래일)

#### ★ 기존 18개 (자동, R1로 다시 계산, 2판 정의 그대로)

| 피처 | 정의 |
|---|---|
| Change | 수정종가 기준 t−1 → t 수익률 |
| Vol20 | 최근 20거래일 일별 로그수익률의 표준편차 |
| VolumeRatio | 거래량 ÷ 20거래일 평균 거래량(주식 단위 변경 보정) |
| Disp20, Disp60 | 수정종가 ÷ 20·60거래일 평균 수정종가 |
| RSI14 | 100 × G ÷ (G + L). G, L은 14거래일 평균 상승폭과 하락폭(수정종가 차분의 단순 평균) |
| LogMarcap | log(시총) |
| Up | Change > 0이면 1, 아니면 0 |
| Ret5 | 수정종가_t ÷ 수정종가_{t−5} − 1 |
| Mom12_1 | 수정종가_{t−21} ÷ 수정종가_{t−252} − 1 |
| High52 | 수정종가 ÷ 최근 252거래일 최고 수정종가 |
| Issuance | log(시총_t ÷ 시총_{t−252}) − log(수정종가_t ÷ 수정종가_{t−252}) |
| Range | (고가 − 저가) ÷ 종가 (당일 비수정 가격) |
| CloseLoc | (종가 − 저가) ÷ (고가 − 저가). 고가 = 저가이면 결측 |
| Gap | 시가 ÷ (종가 ÷ (1 + Change)) − 1 |
| LimitHit | Change ≥ 제한폭 − 0.5%p이면 +1, Change ≤ −제한폭 + 0.5%p이면 −1, 그 외 0 |
| Turnover | 거래대금 ÷ 시총 |
| Amihud | \|Change\| ÷ (거래대금 ÷ 1억 원). 실험에서는 log(Amihud + 1e−8)로 쓴다 |

#### 신규

| 피처 | 정의 | 원천 | 우선 | 근거 |
|---|---|---|---|---|
| ForeignNetBuy, InstNetBuy, IndivNetBuy | 투자자별 순매수대금_t ÷ 시총_t | R6 | ★ | 김수경·변영태(2011), Choe, Kho & Stulz (1999) |
| ForeignNetBuy5·20, InstNetBuy5·20 | 최근 5·20거래일 순매수대금 합 ÷ 시총_t | R6(자동 계산) | ○ | |
| Beta60 | 최근 60거래일 Change를 KOSPI 일별 수익률에 회귀한 기울기. 유효 관측이 40일 미만이면 결측 | R1·R2(자동) | ○ | Ang et al. (2006) |
| IdioVol60 | 위 회귀 잔차의 표준편차 | R1·R2(자동) | ○ | Ang et al. (2006) |
| MAX20 | 최근 20거래일 Change 최댓값 | R1(자동) | ○ | Bali, Cakici & Whitelaw (2011) |
| Skew60 | 최근 60거래일 Change의 표본 왜도 | R1(자동) | ○ | |
| ForeignRatio | 외국인 보유 주식수 ÷ 상장주식수 | R14 | ○ | |
| ForeignExhaust | 외국인 보유 주식수 ÷ 외국인 한도 주식수 | R14 | ○ | |
| dForeignRatio5 | ForeignRatio_t − ForeignRatio_{t−5} | R14 | ○ | |
| ShortRatio | 공매도 거래대금 ÷ 거래대금 | R15 | ○ | Boehmer, Jones & Zhang (2008) |
| ShortBalance | 공매도 잔고금액 ÷ 시총 (2.1절 공시 시차 적용) | R15 | ○ | |
| ShortAllowed | 그날 그 종목의 공매도 허용 여부(2.2절 규제 구간과 KOSPI200 편입 여부로 만듦) | R17(자동) | ○ | |
| K200Member | KOSPI200 편입 종목이면 1 | R17 | ○ | 지수 편입 효과 |
| K200Pending | 편입·편출 발표일부터 적용일 전날까지 +1(편입) 또는 −1(편출) | R17 | ○ | |
| 이벤트 더미 | 배당락일, 권리락일, 증자·분할 적용일, 관리종목, 투자경고·위험, 거래정지 직후 재개일 | R18 | ○ | |
| 기관 세부 순매수 | 6분류별 순매수대금 ÷ 시총 | R25 | △ | |
| MarginBalance | 종목별 신용융자 잔고 ÷ 시총 | R26 | △ | |

### 4.2 재무 GRU 입력 (기업 × 분기, 최근 8분기)

#### 분기 값 만드는 규칙 (자동)

1. **보고서 코드:** 1분기 11013, 반기 11012, 3분기 11014, 사업보고서 11011.
2. **연결과 별도:** 연결을 쓰고, 연결이 없을 때만 별도를 쓴다. 어느 쪽을 썼는지 `FsDiv` 열에 남긴다.
3. **손익·현금흐름 분기 값**
   - 1~3분기 손익은 보고서의 3개월 값을 쓴다.
   - 4분기 손익은 연간 값 − 1~3분기 누적이다.
   - 현금흐름은 누적 값만 나오므로 앞 분기 누적을 빼서 분기 값을 만든다.
4. **재무상태표:** 분기말 값을 그대로 쓴다.
5. **TTM:** 손익·현금흐름은 최근 4개 분기 합(TTM)을 함께 만든다. 4개 분기가 다 없으면 결측.
6. **사용 시작일:** `AvailableFrom` = R5에서 찾은 그 보고서의 **최초** 접수일 다음 거래일.
7. **정정공시:** API 값이 정정 뒤 값일 수 있다. 정정 여부와 정정 접수일을 저장하고, 보고서에 기록한다.
8. **오래된 자료:** 분기말에서 8개월이 지나도록 새 보고서가 없으면 결측으로 바꾼다(`PREPROCESSING_SPEC.md` 3.2절).
9. **금융업:** 업종(R7)이 은행·증권·보험·기타금융이면 `FinancialFlag` = 1. 금융업은 매출 기준 비율(SP, OPM, GPA, CurrentRatio)을 결측으로 둔다.

#### ★ 원계정 (R4에 모두 들어 있음. 별도 수집 불필요)

| 계정 | 대표 account_id (회사마다 다르면 account_nm으로 보완) |
|---|---|
| 매출액 | ifrs-full_Revenue |
| 매출총이익 | ifrs-full_GrossProfit |
| 영업이익 | dart_OperatingIncomeLoss |
| 당기순이익(지배) | ifrs-full_ProfitLossAttributableToOwnersOfParent |
| 당기순이익(전체) | ifrs-full_ProfitLoss |
| 영업활동현금흐름 | ifrs-full_CashFlowsFromUsedInOperatingActivities |
| 총자산 | ifrs-full_Assets |
| 총부채 | ifrs-full_Liabilities |
| 자본총계(지배) | ifrs-full_EquityAttributableToOwnersOfParent |
| 유동자산, 유동부채 | ifrs-full_CurrentAssets, ifrs-full_CurrentLiabilities |
| 현금성자산 | ifrs-full_CashAndCashEquivalents |
| 이익잉여금 | ifrs-full_RetainedEarnings |
| 배당금 지급(현금흐름표) | ifrs-full_DividendsPaidClassifiedAsFinancingActivities |
| 유형자산 취득(CAPEX) | ifrs-full_PurchaseOfPropertyPlantAndEquipment |

#### 파생 피처

일별 표에는 그날 쓸 수 있는 최신 값을 넣는다. 2단계 선형 검사가 이 값을 쓴다. 재무 GRU는 8분기 시퀀스를 쓴다.

| 피처 | 정의 | 우선 | 근거 |
|---|---|---|---|
| BM | 지배자본 ÷ 기업 시총_t (자본 ≤ 0이면 결측) | ★ | Fama & French (1992) |
| EP | TTM 지배순이익 ÷ 기업 시총_t | ★ | Welch & Goyal (2008) |
| SP | TTM 매출 ÷ 기업 시총_t | ★ | |
| CFP | TTM 영업현금흐름 ÷ 기업 시총_t | ★ | |
| ROE | TTM 지배순이익 ÷ (지배자본_q + 지배자본_{q−4}) / 2 | ★ | |
| ROA | TTM 순이익 ÷ (총자산_q + 총자산_{q−4}) / 2 | ★ | |
| OPM | TTM 영업이익 ÷ TTM 매출 | ★ | |
| SalesGrowth | 분기 매출_q ÷ 매출_{q−4} − 1 (분모 ≤ 0이면 결측) | ★ | |
| OIGrowth | (분기 영업이익_q − 영업이익_{q−4}) ÷ 총자산_{q−4} | ★ | 분모를 자산으로 해서 적자 전환 때 부호가 뒤집히지 않게 함 |
| Leverage | 총부채 ÷ 총자산 | ★ | |
| CurrentRatio | 유동자산 ÷ 유동부채 | ★ | |
| CashToAssets | 현금성자산 ÷ 총자산 | ★ | |
| FinancialFlag | 금융업이면 1 | ★ | |
| DY | TTM 배당금 지급 ÷ 기업 시총_t | ○ | 지수 배당수익률의 예측력 |
| GPA | TTM 매출총이익 ÷ 총자산 | ○ | Novy-Marx (2013) |
| AssetGrowth | 총자산_q ÷ 총자산_{q−4} − 1 | ○ | Cooper, Gulen & Schill (2008) |
| Accruals | (TTM 순이익 − TTM 영업현금흐름) ÷ 평균 총자산 | ○ | Sloan (1996), Hirshleifer, Hou & Teoh (2009) |
| SUE | (분기 순이익_q − 순이익_{q−4}) ÷ 최근 8개 분기 같은 변화의 표준편차(최소 6개) | ○ | Bernard & Thomas (1989) |
| DaysSinceFiling | 최근 정기보고서 사용 시작일 이후 경과 거래일 | ○ | |
| FilingDay | 정기보고서 사용 시작일 당일이면 1 | ○ | Savor & Wilson (2016) |
| PrelimDay, PrelimOIGrowth | 잠정실적 사용 시작일 더미, 잠정 영업이익 전년 동기 대비 변화 ÷ 총자산 | ○ | 대형주 잠정실적이 정기보고서보다 먼저 나옴 |
| CapexToAssets | TTM CAPEX ÷ 총자산 | △ | |

### 4.3 GNN 입력: 기업 간 관계

| 관계 | 정의 | 원천 | 우선 |
|---|---|---|---|
| 동일 업종 | 그날 적용되는 KRX 업종이 같은 기업끼리 연결. 기준일 사이에는 직전 기준일의 분류를 쓴다 | R7 | ★ |
| 60일 수익률 상관 이웃 | 최근 60거래일 Change 상관 상위 10개 이웃(최소 40일, 무방향, 자기 연결 포함) | R1(자동, 2판 `CorrelationGraphs`와 같은 규칙) | ★ |
| 기업집단 | 같은 해 공정위 지정 집단 소속끼리 연결 | R20 | ○ |
| 세부 업종 | WICS·GICS 중분류 | R27 | △ |
| 공급망 | 주요 고객·공급사 | R28 | △ |

### 4.4 시장 GRU 입력 (일별, 최근 60거래일)

#### ★ 기존 7개 (자동, 2판 정의 그대로)

| 피처 | 정의 |
|---|---|
| IndexReturn | KOSPI t−1 → t 수익률 |
| IndexVol20 | KOSPI 20거래일 일별 수익률 표준편차 |
| IndexMom20, IndexMom60 | KOSPI 20·60거래일 수익률 |
| AvgVar | 유효 종목 Vol20²의 동일가중 평균 (Goyal & Santa-Clara 2003) |
| AvgCorr | IndexVol20² ÷ (시총가중 평균 Vol20)² (Pollet & Wilson 2010) |
| Breadth | 유효 종목 중 상승 종목 비율 |

#### 국내 일별 신규

| 피처 | 정의 | 원천 | 우선 | 근거 |
|---|---|---|---|---|
| KTB3Y, KTB10Y, CD91 | 금리 수준(%) | R10 | ★ | Fama & French (1989) |
| dKTB3Y, dKTB10Y | 금리 일별 변화(%p) | R10 | ★ | |
| Term | KTB10Y − KTB3Y | R10 | ★ | Fama & French (1989) |
| Credit | CorpAAm3Y − KTB3Y | R10 | ★ | Fama & French (1989) |
| FXRet | log(USDKRW_t ÷ USDKRW_{t−1}) | R11 | ★ | |
| FXVol20 | FXRet 20거래일 표준편차 | R11 | ★ | |
| VKOSPI | VKOSPI 종가 | R9 | ★ | |
| VRP | VKOSPI − 100 × √252 × IndexVol20 | R9 | ★ | Bollerslev, Tauchen & Zhou (2009) |
| MktForeignNet, MktInstNet, MktIndivNet | 시장 전체 순매수대금 ÷ 코스피 전체 시총 | R8 | ★ | |
| BaseRate | 그날 적용되는 기준금리 | R12 | ★ | |
| RateDirection | 가장 최근 변경의 방향(+1 인상, −1 인하) | R12 | ★ | |
| MPCDay | 금통위 결정일이면 1 | R12 | ★ | |
| ShortBan | 공매도 전면 금지 구간이면 1 | 2.2절 | ★ | |
| MktForeignNet5·20 등 | 시장 순매수 5·20일 합 ÷ 시총 | R8(자동) | ○ | |
| MktShortRatio, MktShortBalance | 시장 공매도 거래 비중, 잔고 ÷ 시총 | R16 | ○ | Rapach, Ringgenberg & Zhou (2016) |
| Basis | (K200Futures − K200Spot) ÷ K200Spot | R21 | ○ | |
| dOpenInterest, ForeignFuturesNet | 미결제약정 변화율, 외국인 선물 순매수 | R21 | ○ | |
| ProgramArb, ProgramNonArb | 프로그램 차익·비차익 순매수 ÷ 시총 | R21 | ○ | |
| DepositsGrowth20, MarginGrowth20 | 고객예탁금·신용융자 20일 변화율 | R22 | ○ | Baker & Wurgler (2006) |
| JPYRet, CNYRet | 원/엔, 원/위안 일별 로그수익률 | R29 | △ | |

#### 해외 (★·○, 정확한 날짜 규칙)

- u(t)는 미국 날짜가 한국 날짜 t보다 엄격히 앞선 미국 거래일 가운데 가장 최근 날이다.
- 수익률 피처는 ln(가격_{u(t)} ÷ 가격_{u(t′)})이다. t′는 직전 한국 거래일이다. 그 사이 미국이 쉬었으면 0이다.
- 수준 피처는 u(t)의 값을 쓴다.
- 점검을 위해 일별 표에 `USDateUsed` = u(t) 열을 남긴다.

| 피처 | 원천 | 우선 | 근거 |
|---|---|---|---|
| SPXRet, NASDAQRet | R13 | ★ | Rapach, Strauss & Zhou (2013) |
| SOXRet | R13 | ★ | 코스피의 반도체 비중 |
| VIX, dVIX | R13 | ★ | |
| US10Y, Term US(10년 − 2년), DXYRet | R23 | ○ | |
| WTIRet, CopperRet, GoldRet | R23 | ○ | Driesprong, Jacobsen & Maat (2008) |
| NikkeiRet, ShanghaiRet, HSIRet (t−1일) | R30 | △ | |
| FOMCDay, USCPIDay, USJobsDay(한국 다음 거래일) | R31 | △ | |

#### 월별 거시 (○, 발표일 다음 거래일부터 다음 발표 전까지 같은 값)

| 피처 | 원천 | 발표일을 모를 때 쓰는 보수적 날짜 |
|---|---|---|
| CPI 전년비, 근원 CPI 전년비 | R24 | 다음 달 10일 |
| 수출 전년비 | R24 | 다음 달 2일 |
| 20일 잠정수출 전년비 | R24 | 당월 22일 |
| 실업률, 취업자 증감 | R24 | 다음 달 20일 |
| 산업생산 전년비, 경기선행지수 순환변동치 | R24 | 다음 달 말일 다음 날 |
| 소비자심리지수, 기업경기실사지수 | R24 | 다음 달 1일 |

### 4.5 달력 (○, 자동 생성)

| 피처 | 정의 |
|---|---|
| OptionExpiry | 매월 둘째 목요일(휴장이면 직전 거래일)이면 1 |
| QuadWitching | 3·6·9·12월의 OptionExpiry이면 1 |
| K200Rebalance | 6·12월 선물 만기일 다음 거래일(정기변경 적용일)이면 1 |
| MonthEnd, QuarterEnd | 월·분기 마지막 거래일이면 1 |
| PreHoliday | 다음 거래일까지 평일 휴장이 하루 이상 끼면 1 |
| PostHoliday | 직전 거래일과의 사이에 평일 휴장이 하루 이상 끼면 1 |
| EarningsSeason | 분기말 다음 날부터 45일 안(4분기는 90일 안)이면 1 |
| DayOfWeek | 요일(원-핫) |

---

## 5. 타깃, 마스크, 기업 집합 (★, 자동, 기존 규칙)

| 항목 | 정의 |
|---|---|
| NextDayIndexReturn | KOSPI_{t+1} ÷ KOSPI_t − 1 (주 타깃) |
| NextDayReturn | t+1 수정종가 수익률. 다음 조건을 모두 만족할 때만 값이 있다: t+1일에 그 종목 행이 있음, 가격 연결이 끊기지 않음, t+1일 거래량 > 0, 공식 상장폐지일이 앞으로 10거래일 안에 있지 않음. 그 외에는 결측이고, 이유를 `NextDayStatus`에 적는다 |
| 보통주 판정 | 코드 끝자리 0을 보통주로 본다. 상장폐지 기준표에 종류(Kind = 보통주, SecuGroup ∈ {주권, 외국주권})가 있으면 그것을 우선한다. 1절 제외 목록은 이름 규칙으로 뺀다 |
| Valid | 보통주, 거래량 > 0, 기본 7개 피처(Change, Vol20, VolumeRatio, Disp20, Disp60, RSI14, LogMarcap)가 모두 유한, 시총 > 0 |
| 기업 집합 U_t | Valid이고, 직전 60거래일(t 포함) 중 Valid인 날이 40일 이상 |
| CapWeight | U_t 안의 시총 비중 |
| CapPct | U_t 안의 시총 백분위 순위(가장 큰 종목 1, 가장 작은 종목 0) |
| FinValid, FlowValid | BM·EP·SP가 모두 있음 / ForeignNetBuy가 있음 |

---

## 6. 최종 산출물 (자동, `rev4/data/`)

| 파일 | 행 단위 | 내용 |
|---|---|---|
| `firms_daily.parquet` | (Date, Code) | 5절 열, 4.1절 거래 피처, 4.2절 재무 최신 값, SectorID, GroupID |
| `firms_quarterly.parquet` | (Code, FiscalPeriodEnd) | ReportType, FsDiv, RceptNo, FirstRceptDt, AvailableFrom, Corrected, 원계정 분기 값, TTM, 파생 피처 |
| `market_daily.parquet` | Date | KOSPI, NextDayIndexReturn, 4.4절 시장 피처, 4.5절 달력, USDateUsed |
| `relations.parquet` | (ValidFrom, Code) | ValidTo, Sector, Group |
| `feature_coverage.csv` | 피처 | 연도별 U_t 커버리지, 첫 50% 도달 연도, 정의, 근거 |
| `audit_*.csv` | | 7절 점검 결과 |
| `manifest.csv` | 자료 | 2.2절 수집 기록 |

---

## 7. 데이터 관문 (4판 1단계): 하나라도 어긋나면 2단계로 가지 않는다

1. **기존 데이터 재현**
   - 다시 만든 18개 피처와 NextDayReturn이 기존 `daily_dataset/daily_firms.parquet`과 같은 (Date, Code)에서 99.9% 이상 일치해야 한다(상대오차 1e−6).
2. **지수 재현**
   - 전날 종가 시총으로 가중한 유효 종목 수익률과 KOSPI 수익률의 상관이 2010년대와 2020년대 모두 0.98 이상이어야 한다.
3. **타깃 완전성**
   - 2015-01-02 ~ 2026-10-29 모든 거래일에 NextDayIndexReturn이 있어야 한다.
   - 2015년 이후 하루 유효 종목 수의 최솟값이 500 이상이어야 한다.
4. **시점 점검** (전 행 검사)
   - 재무: AvailableFrom > FirstRceptDt
   - 미국: USDateUsed < Date
   - 월별 거시: 사용 시작일 > 발표일
   - 공매도 잔고: 사용일 ≥ 기준일 + 시차
5. **손 대조**
   - 대상: 시총 상위 10개 종목, 상장폐지 종목 5개
   - 각 4개 분기의 매출, 영업이익, 지배자본, 접수일을 DART 원문과 대조한다.
6. **커버리지**
   - 신규 피처마다 2015~2021년 U_t 커버리지를 연도별로 보고한다.
   - 금융업 결측은 빼고 계산한다.
   - 평균 70% 미만이면 2단계 주 입력에서 빼고, 이유를 기록한다.
7. **생존편향**
   - 2015~2026년에 상장폐지된 코스피 보통주 가운데 R4 재무와 R6 수급이 있는 비율을 보고한다.

---

## 8. 수집 순서와 분담

### 1차: ★ 필수 (사용자)

| 할 일 | 필요한 것 | 메모 |
|---|---|---|
| R4·R5 DART 재무와 공시 목록 | OpenDART API 키(무료) | 키를 환경변수 `DART_API_KEY`로 설정해 주면 Claude가 받을 수 있다. 일일 호출 한도가 있어(개인 키 약 2만 건) 여러 날 걸릴 수 있다. DART 웹의 "재무정보 일괄다운로드"가 더 빠를 수 있다 |
| R6·R7·R8·R9 KRX 수급·업종·VKOSPI | KRX 정보데이터시스템 회원 계정 | pykrx는 이제 로그인이 필요하다. 환경변수 `KRX_ID`, `KRX_PW`를 직접 설정하면 `code/download_investor_flows.py`가 그대로 돈다. **비밀번호는 채팅에 쓰지 않는다** |
| R10·R11·R12 ECOS 금리·환율·기준금리 | ECOS API 키(무료) | 환경변수 `ECOS_API_KEY`로 설정해 주면 Claude가 받을 수 있다. ECOS 웹에서 CSV로 받아도 된다 |

**먼저 확인할 것:** OpenDART 재무 API가 몇 년도 자료부터 주는지 확인한다.
- 2015 사업연도부터라는 안내가 흔하지만, 이번에 공식 근거로 확인하지는 못했다.
- 2011~2014년이 없으면 일괄다운로드나 유료 DB(TS2000, FnGuide)로 보완하거나, 결측으로 두고 재무 창이 덜 찬 구간을 마스크로 처리한다.
- 어느 쪽인지는 2단계 시작 전에 정한다.

### 같은 시기: 자동 (Claude)

- R1 시세를 다시 받는다.
- R13 미국 지수를 받는다.
- 기존 18 + 7개 피처를 다시 계산하고 기존 파일과 대조한다(관문 1~3).
- 위험·분포 피처, 상관 이웃, 달력, 타깃, 마스크를 만든다.

### 2차: ○ 권장

- R14~R24를 모은다.
- R23은 자동으로 받는다.

### 3차: △ 선택

- R25~R31은 2단계 선형 검사 결과를 본 뒤에 정한다.
