# 4판 전처리 명세 (확정안)

작성 2026-11-20. 수집 명세 `FEATURE_COLLECTION_LIST.md`의 원천 자료(R1~R31)를 모델 입력으로 바꾸는 규칙이다. 데이터를 받으면 이 문서대로 처리하고, 바꿀 일이 생기면 7절에 기록한 뒤 바꾼다.

## 0. 원칙

1. **시점:** t일 피처에는 t일 15:30 장 마감 시점에 공개된 값만 쓴다. 발표 시각을 모르면 다음 거래일부터 쓴다.
   - 재무: Hou, Xue & Zhang (2020)처럼 실제 공시일을 기준으로 삼는다. Fama & French (1992)의 6개월 고정 시차는 쓰지 않는다.
   - 거시 지표: Croushore & Stark (2001)처럼 실시간(최초 발표) 값을 쓴다.
2. **학습 구간 통계만 쓴다.** 윈저화 경계, 평균, 표준편차, 추세 제거 계수는 재학습(F⁽ᵏ⁾)마다 그 학습 구간에서만 다시 계산한다. 전체 표본 통계는 미래 정보다(Welch & Goyal 2008).
3. **시계열 수준을 보존한다. 날짜별 순위 변환을 하지 않는다.**
   - Gu, Kelly & Xiu (2020)의 횡단면 순위 변환은 기업 간 상대 수익률을 예측할 때 맞는 방법이다.
   - 이 연구는 기업 예측을 합쳐 지수를 예측한다. 지수 예측력은 기업 특성의 **시계열 수준**에서 나온다(Kelly & Pruitt 2013, Pontiff & Schall 1998). 날마다 순위로 바꾸면 그 정보가 사라진다.
   - 그래서 학습 구간 전체를 한데 모은 분포로 윈저화하고 표준화한다.
4. **결측:**
   - 표준화한 뒤 0(학습 평균)으로 채운다. 결측을 횡단면 중앙값으로 채우는 Gu, Kelly & Xiu (2020)의 방식과 같은 취지다.
   - 자료 묶음마다 결측 표시 변수를 같이 넣는다. 그래야 모델이 "0이라는 값"과 "자료 없음"을 구분한다.
5. **생존편향:** 상장폐지 종목을 포함한다. 상장폐지 직전 구간은 타깃에서 빼고 이유를 기록한다(Shumway 1997).

## 1. 처리 순서 (모든 피처 공통)

1. 원천 자료를 시점에 맞춰 정렬한다(2.1절, 3.2절, 5절의 규칙).
2. 원래 단위로 피처를 계산한다.
3. 변환한다: 아래 표의 log 같은 변환. 지정이 없으면 원래 단위를 쓴다.
4. 이산 지표(학습 구간에서 고유값 3개 이하)는 평균만 뺀다. 윈저화와 표준편차 나눗셈을 하지 않는다(3판 D1 규칙).
5. 연속 피처는 학습 구간의 기업 집합 U_t 행을 모두 모아 1%, 99% 분위수로 자른다(Green, Hand & Zhang 2017의 1%/99% 윈저화).
6. 학습 구간 평균과 표준편차로 z-score를 만든다.
7. 결측은 0으로 채우고, 묶음별 결측 표시를 붙인다.
8. 시장 피처는 윈저화하지 않는다. 표준화한 뒤 ±5에서 자른다(우리 규칙). 폭락일 정보를 지우지 않으면서 신경망 학습이 극단값에 흔들리지 않게 한다.

## 2. 기업 일별 피처 (거래 GRU 입력, 60거래일 창)

### 2.1 시점

- 시세, 수급, 공매도 거래는 t일 값을 쓴다.
- 공매도 잔고는 기준일에 공시 시차(보통 2거래일)를 더한 날부터 쓴다.
- 외국인 보유는 t일 값을 쓴다.

### 2.2 피처별 변환

| 피처 | 변환 | 근거 |
|---|---|---|
| 기존 18개 | 2판 그대로. Amihud만 log(Amihud + 1e−8). Up, LimitHit는 이산 지표 | Amihud (2002) |
| 투자자 순매수 ÷ 시총 (당일, 5일 합, 20일 합) | 원래 단위, 1%/99% 윈저화 | Choe, Kho & Stulz (1999) |
| 외국인 보유 비중, 한도 소진율 | 원래 단위(0~1) | |
| 외국인 보유 비중 5일 변화 | 원래 단위, 윈저화 | |
| 공매도 거래 비중 | 원래 단위, 윈저화. **공매도 금지 구간과 공매도 불가 종목은 결측으로 두고** `ShortBan` 표시를 붙인다(금지 중의 0은 정보가 아니라 제도의 결과) | Boehmer, Jones & Zhang (2008) |
| 공매도 잔고 ÷ 시총 | log(1 + 100 × 값), 윈저화. 결측 처리는 위와 같음 | |
| 베타 | 60일, Dimson 보정: 당일과 전날 시장 수익률에 회귀한 두 계수의 합. 관측 40일 미만이면 결측 | Dimson (1979). 거래가 드문 소형주의 베타 하향 편의를 줄임 |
| 고유 변동성 | 위 회귀 잔차의 표준편차, log | Ang, Hodrick, Xing & Zhang (2006) |
| MAX | 최근 20거래일 최대 일수익률 | Bali, Cakici & Whitelaw (2011) |
| 왜도 | 60일 표본 왜도 | |
| KOSPI200 편입, 편입·편출 예정, 이벤트 더미 | 이산 지표(평균만 뺌) | |

### 2.3 결측 표시

- `FlowMissing`: 수급 자료가 없는 날
- `ShortMissing`: 공매도 자료가 없거나 금지 구간인 날

## 3. 재무 피처

### 3.1 분기 값 만들기

1. 연결재무제표를 우선하고, 없으면 별도재무제표를 쓴다. 사용한 쪽을 `FsDiv`에 기록한다.
2. 손익
   - 1~3분기는 3개월 값을 쓴다.
   - 4분기는 연간 값 − 1~3분기 누적이다.
3. 현금흐름은 누적 값의 차분으로 분기 값을 만든다.
4. TTM은 최근 4개 분기의 합이다. 하나라도 없으면 결측이다.
5. 계정 매칭
   - `account_id`로 먼저 찾는다.
   - 없으면 `account_nm` 동의어 목록(예: 매출액, 수익(매출액), 영업수익)으로 찾는다.
   - 매칭률을 계정별로 보고한다.

### 3.2 시점과 유효기간

- **사용 시작일**은 최초 접수일 다음 거래일이다(Hou, Xue & Zhang 2020의 공시일 기준).
- **유효기간**: 가장 최근 분기말에서 **8개월**이 지나면 결측으로 바꾼다.
  - Hou, Xue & Zhang (2020)은 6개월을 쓴다.
  - 한국 사업보고서 기한(결산 후 90일) 때문에 3분기말부터 사업보고서 제출까지 6개월이 걸린다. 그래서 2개월 여유를 둔다.
- **정정공시**: 정정된 값은 정정 접수일 다음 거래일부터 쓴다. API가 정정 뒤 값만 주면 최초 접수일부터 정정값을 쓰게 된다. 이 경우를 `Corrected` 표시로 기록하고 보고서에 한계로 적는다.

### 3.3 재무 GRU 입력: 회계 비율 시퀀스 (가격과 무관한 값만)

- 각 날짜 t에서 그때 사용할 수 있는 최근 **8개 분기**를 시간 순서로 넣는다.
- 분기마다 넣는 벡터는 다음과 같다.

| 피처 | 정의·변환 | 근거 |
|---|---|---|
| ROE, ROA | TTM 이익 ÷ 평균 자본·자산 | |
| 영업이익률, GP/A | TTM 기준. 금융업은 결측 | Novy-Marx (2013) |
| 매출 증가율 | 전년 동기 대비. 분모 ≤ 0이면 결측 | |
| 영업이익 변화 ÷ 자산 | (분기 영업이익 − 전년 동기) ÷ 전년 동기 총자산 | 적자 전환 때 부호가 뒤집히지 않게 함 |
| 자산 증가율 | 전년 동기 대비 | Cooper, Gulen & Schill (2008) |
| 발생액 | (TTM 순이익 − TTM 영업현금흐름) ÷ 평균 자산. 현금흐름표 방식 | Sloan (1996), Hribar & Collins (2002) |
| SUE | (분기 순이익 − 전년 동기) ÷ 최근 8개 분기 같은 변화의 표준편차(최소 6개). 계절 랜덤워크 기대 | Foster, Olsen & Shevlin (1984), Bernard & Thomas (1989) |
| 부채비율, 유동비율, 현금 ÷ 자산 | 총부채 ÷ 총자산 등. 유동비율은 금융업이면 결측 | |
| CAPEX ÷ 자산 | 선택(△) | |
| log(총자산) | 규모 | |
| 분기 결측 표시, 사용 시작 후 경과 거래일 | 분기가 없으면 벡터를 0으로 두고 표시는 1 | |
| 금융업 표시 | 이산 지표 | |

- 원계정 금액(원)은 직접 넣지 않는다. 기업 규모에 따라 단위가 수천 배 달라서 비율과 증가율만 쓴다.
- 표준화 통계는 학습 구간의 (기업, 분기) 행을 모아 계산한다.

### 3.4 가격을 쓰는 가치 비율: 일별 피처로 거래 GRU에 넣는다

- 분모는 **t일의 기업 시총**(보통주와 우선주 합)이다.
- Asness & Frazzini (2013)는 6개월 지난 가격보다 현재 가격을 쓴 가치 비율이 더 정확하다고 보였다.
- 분자는 3.2절 규칙으로 그날 쓸 수 있는 최신 재무 값이다.

| 피처 | 변환 | 근거 |
|---|---|---|
| BM | log(지배자본 ÷ 기업 시총). 자본 ≤ 0이면 결측 | Fama & French (1992) |
| EP | TTM 지배순이익 ÷ 기업 시총, 원래 단위. **EP < 0이면 1인 표시를 따로 둔다** | Fama & French (1992)의 E/P(+)와 음수 더미 |
| SP | log(TTM 매출 ÷ 기업 시총). 금융업은 결측 | |
| CFP | TTM 영업현금흐름 ÷ 기업 시총 | |
| DY | TTM 배당금 지급 ÷ 기업 시총 (0 포함) | |
| 공시 이벤트 | 공시 당일 표시, 공시 후 경과 거래일, 잠정실적 당일 표시 | Savor & Wilson (2016) |
| `FinMissing` | 재무 자료가 없거나 유효기간이 지난 날 | |

## 4. 기업 간 관계 (GNN)

| 관계 | 처리 |
|---|---|
| 동일 업종 | 각 기준일의 KRX 업종을 다음 기준일 전날까지 쓴다. 같은 날 U_t 안의 같은 업종 기업끼리 연결하고 자기 연결을 포함한다. 업종이 없는 종목은 자기 연결만 둔다(Hou 2007) |
| 60일 수익률 상관 이웃 | t일까지의 60거래일 Change로 상관을 계산하고 상위 10개 이웃을 고른다(최소 40일, 무방향). 2판 규칙 그대로 |
| 기업집단 | 매년 5월 지정 결과를 다음 거래일부터 1년간 쓴다 |

- 업종 분류가 한 시점분만 있으면 과거에 적용할 때 약한 미래 정보가 섞인다. 받은 기준일 목록을 기록하고, 결과에 한계로 적는다.

## 5. 시장 피처 (시장 GRU 입력, 60거래일 창)

- **원칙:** 수준이 매우 오래 지속되는 변수를 그대로 넣으면 예측 회귀에 편의가 생긴다(Stambaugh 1999). 그런 변수는 변화량, 스프레드, 이동평균 대비 값으로 바꾼다.

| 묶음 | 피처 | 변환 | 근거 |
|---|---|---|---|
| 기존 7개 | IndexReturn 등 | 2판 그대로 | |
| 단기 금리 | 상대 금리 = CD91 − 최근 250거래일 평균 | 수준 대신 이동평균 대비 | Campbell (1991), Ang & Bekaert (2007) |
| 장단기·신용 | 장단기 금리차(국고 10년 − 3년), 신용 스프레드(회사채 AA− − 국고 3년) | 수준 | Fama & French (1989) |
| 금리 변화 | 국고 3년·10년 일별 변화(%p) | | |
| 기준금리 | 12개월 변화, 최근 변경 방향(+1, 0, −1), 금통위 결정일 표시 | 수준은 넣지 않음 | |
| 환율 | 원/달러 일별 로그수익률, 20일 변동성 | | |
| 변동성 | log(VKOSPI), 분산 위험 프리미엄 = (VKOSPI² − (100 × √252 × IndexVol20)²) ÷ 100 | | Bollerslev, Tauchen & Zhou (2009) |
| 시장 수급 | 외국인·기관·개인 순매수 ÷ 코스피 시총(당일, 5일 합, 20일 합) | | |
| 시장 공매도 | log(잔고 ÷ 시총). 추세는 **학습 구간 안에서만** 선형으로 제거 | 원 논문은 전체 표본으로 추세를 뺐다. 이는 미래 정보라서 우리는 학습 구간만 쓴다 | Rapach, Ringgenberg & Zhou (2016) |
| 투자 심리 | 고객예탁금·신용융자 20일 변화율 | | Baker & Wurgler (2006) |
| 선물 | 베이시스, 미결제약정 변화율, 외국인 선물 순매수 ÷ 시총 | | |
| 미국(t−1일) | S&P 500·나스닥·SOX 수익률(직전 한국 거래일 이후 누적), log(VIX), VIX 변화 | 날짜 규칙은 수집 명세 4.4절 | Rapach, Strauss & Zhou (2013) |
| 해외 금리·원자재 | 미 10년 − 2년, 미 10년 변화, 달러지수·유가·구리·금 수익률 | | Driesprong, Jacobsen & Maat (2008) |
| **시장 밸류에이션(신규, 직접 계산)** | log(Σ TTM 배당 ÷ Σ 시총), log(Σ TTM 순이익 ÷ Σ 시총), log(Σ 지배자본 ÷ Σ 시총). U_t 안 기업의 합계 비율 | 별도 수집 없이 재무 자료로 계산 | Welch & Goyal (2008), Pontiff & Schall (1998), Lewellen (2004) |
| 월별 거시 | 물가·수출·산업생산은 전년비, 실업률은 전년 대비 변화, 경기선행 순환변동치 − 100, 소비자심리·기업경기실사 − 100 | 발표일 다음 거래일부터 다음 발표 전까지 같은 값 | Croushore & Stark (2001) |
| 달력 | 만기일, 정기변경, 월말, 연휴 전후, 실적 시즌, 요일 | 이산 지표 | |

시장 밸류에이션은 평균의 비율이 아니라 **합계의 비율**로 만든다. 그래야 시총 가중 지수 수준의 배당수익률과 같아진다.

## 6. 타깃과 기업 집합

- 타깃, 보통주 판정, Valid, U_t, CapWeight, CapPct는 수집 명세 5절과 같다(2판 규칙).
- 기업 다음 날 수익률은 **손실 계산에만** 학습 구간 1%/99% 윈저화를 적용한다. 평가와 지수 합산에는 원래 값을 쓴다.
- 지수 타깃 NextDayIndexReturn과 기업 타깃은 100을 곱해 %로 쓴다.

## 7. 누수 점검 (전처리 직후, 2단계 전에)

1. 재무 피처: 전 행에서 사용 시작일 > 최초 접수일.
2. 미국 피처: 전 행에서 사용한 미국 날짜 < 한국 날짜.
3. 월별 거시: 사용 시작일 > 발표일.
4. 공매도 잔고: 사용일 ≥ 기준일 + 시차.
5. 표준화 통계와 추세 계수: 재학습마다 학습 구간 날짜만 썼는지 코드에서 확인하고, 통계표를 저장한다.
6. **하루 밀기 점검:** 모든 피처를 하루 늦춰(t−1 값 사용) 선형 모델의 검증 성능을 다시 잰다. 늦춘 쪽이 크게 나빠지는 피처는 같은 날 정보 누수가 없는지 다시 확인한다.

## 8. 2판 대비 바뀐 점

| 항목 | 2판 | 4판 |
|---|---|---|
| 이산 지표 | 다른 피처와 같이 윈저화·표준화 | 평균만 뺌 |
| 결측 | 0으로 채움 | 0으로 채우고 묶음별 결측 표시 추가 |
| 시장 피처 | 7개, 학습 구간 표준화 | 신규 묶음 추가, 지속성 강한 변수 변환, 표준화 뒤 ±5 자르기 |
| 재무 | 없음 | 회계 비율 시퀀스(재무 GRU)와 가치 비율 일별 피처로 나눔 |

## 참고문헌

- Amihud, Y. (2002). Illiquidity and stock returns. *Journal of Financial Markets*.
- Ang, A., & Bekaert, G. (2007). Stock return predictability: Is it there? *Review of Financial Studies*.
- Ang, A., Hodrick, R., Xing, Y., & Zhang, X. (2006). The cross-section of volatility and expected returns. *Journal of Finance*.
- Asness, C., & Frazzini, A. (2013). The devil in HML's details. *Journal of Portfolio Management*.
- Baker, M., & Wurgler, J. (2006). Investor sentiment and the cross-section of stock returns. *Journal of Finance*.
- Bali, T., Cakici, N., & Whitelaw, R. (2011). Maxing out. *Journal of Financial Economics*.
- Bernard, V., & Thomas, J. (1989). Post-earnings-announcement drift. *Journal of Accounting Research*.
- Boehmer, E., Jones, C., & Zhang, X. (2008). Which shorts are informed? *Journal of Finance*.
- Bollerslev, T., Tauchen, G., & Zhou, H. (2009). Expected stock returns and variance risk premia. *Review of Financial Studies*.
- Campbell, J. (1991). A variance decomposition for stock returns. *Economic Journal*.
- Choe, H., Kho, B.-C., & Stulz, R. (1999). Do foreign investors destabilize stock markets? The Korean experience in 1997. *Journal of Financial Economics*.
- Cooper, M., Gulen, H., & Schill, M. (2008). Asset growth and the cross-section of stock returns. *Journal of Finance*.
- Croushore, D., & Stark, T. (2001). A real-time data set for macroeconomists. *Journal of Econometrics*.
- Dimson, E. (1979). Risk measurement when shares are subject to infrequent trading. *Journal of Financial Economics*.
- Driesprong, G., Jacobsen, B., & Maat, B. (2008). Striking oil: Another puzzle? *Journal of Financial Economics*.
- Fama, E., & French, K. (1989). Business conditions and expected returns on stocks and bonds. *Journal of Financial Economics*.
- Fama, E., & French, K. (1992). The cross-section of expected stock returns. *Journal of Finance*.
- Foster, G., Olsen, C., & Shevlin, T. (1984). Earnings releases, anomalies, and the behavior of security returns. *The Accounting Review*.
- Green, J., Hand, J., & Zhang, X. F. (2017). The characteristics that provide independent information about average U.S. monthly stock returns. *Review of Financial Studies*.
- Gu, S., Kelly, B., & Xiu, D. (2020). Empirical asset pricing via machine learning. *Review of Financial Studies*.
- Hou, K. (2007). Industry information diffusion and the lead-lag effect in stock returns. *Review of Financial Studies*.
- Hou, K., Xue, C., & Zhang, L. (2020). Replicating anomalies. *Review of Financial Studies*.
- Hribar, P., & Collins, D. (2002). Errors in estimating accruals. *Journal of Accounting Research*.
- Kelly, B., & Pruitt, S. (2013). Market expectations in the cross-section of present values. *Journal of Finance*.
- Lewellen, J. (2004). Predicting returns with financial ratios. *Journal of Financial Economics*.
- Novy-Marx, R. (2013). The other side of value: The gross profitability premium. *Journal of Financial Economics*.
- Pontiff, J., & Schall, L. (1998). Book-to-market ratios as predictors of market returns. *Journal of Financial Economics*.
- Rapach, D., Ringgenberg, M., & Zhou, G. (2016). Short interest and aggregate stock returns. *Journal of Financial Economics*.
- Rapach, D., Strauss, J., & Zhou, G. (2013). International stock return predictability: What is the role of the United States? *Journal of Finance*.
- Savor, P., & Wilson, M. (2016). Earnings announcements and systematic risk. *Journal of Finance*.
- Shumway, T. (1997). The delisting bias in CRSP data. *Journal of Finance*.
- Sloan, R. (1996). Do stock prices fully reflect information in accruals and cash flows about future earnings? *The Accounting Review*.
- Stambaugh, R. (1999). Predictive regressions. *Journal of Financial Economics*.
- Welch, I., & Goyal, A. (2008). A comprehensive look at the empirical performance of equity premium prediction. *Review of Financial Studies*.
