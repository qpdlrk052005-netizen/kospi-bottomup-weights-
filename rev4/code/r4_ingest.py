"""Raw collected data -> processed model inputs, following PREPROCESSING_SPEC.md (rev4).

Raw files (FEATURE_COLLECTION_LIST.md section 3) are read from rev4/data/raw/; every block is optional and skipped when
its file is missing. Outputs (rev4/data/processed/, or $R4_PROCESSED):
  firm_daily_extra.parquet    Date, Code, <daily firm features>        (point-in-time, transformed, not scaled)
  market_daily_extra.parquet  Date, <daily market features>
  firm_quarterly.parquet      Code, FiscalPeriodEnd, AvailableFrom, <accounting ratios>
  relations.parquet           ValidFrom, Code, Sector
  audit/*.csv, audit/ingest_report.json
Scaling (winsorising, z-scores) is not done here: r4_common does it inside every training window.

  python -X utf8 rev4/code/r4_ingest.py            # all blocks whose raw files exist
  python -X utf8 rev4/code/r4_ingest.py --derived  # also the price-derived risk features (beta, idio vol, MAX, skew)
"""
import argparse
import json
import os
from pathlib import Path
import re

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REV4 = HERE.parent
ROOT = REV4.parent
RAW = Path(os.environ.get("R4_RAW", REV4 / "data" / "raw"))
OUT = Path(os.environ.get("R4_PROCESSED", REV4 / "data" / "processed"))
DATA_ROOT = Path(os.environ.get("R4_DATA_ROOT", ROOT))      # KOSDAQ build: the kosdaq/ folder
FLOWS_FILE = os.environ.get("R4_FLOWS_FILE", "investor_flows_kospi.csv")

SHORT_BANS = (("2020-03-16", "2021-05-02"), ("2023-11-06", "2025-03-30"))
SHORT_PARTIAL = ("2021-05-03", "2023-11-05")       # only KOSPI200 / KOSDAQ150 names could be shorted
SHORT_BALANCE_LAG = 2                              # trading days between balance date and disclosure
STALE_DAYS = 243
REPORTS = {"11013": 3, "11012": 6, "11014": 9, "11011": 12}       # reprt_code -> fiscal month end (Dec FYE)
ACCOUNTS = {   # name -> (account_id list, account_nm fallbacks, statement kinds)
    "Revenue": (["ifrs-full_Revenue", "ifrs_Revenue"], ["매출액", "수익(매출액)", "영업수익", "매출"], ("IS", "CIS")),
    "GrossProfit": (["ifrs-full_GrossProfit", "ifrs_GrossProfit"], ["매출총이익"], ("IS", "CIS")),
    "OperatingIncome": (["dart_OperatingIncomeLoss"], ["영업이익", "영업이익(손실)"], ("IS", "CIS")),
    "NetIncomeParent": (["ifrs-full_ProfitLossAttributableToOwnersOfParent", "ifrs_ProfitLossAttributableToOwnersOfParent"],
                        ["지배기업의 소유주에게 귀속되는 당기순이익", "지배기업 소유주지분"], ("IS", "CIS")),
    "NetIncome": (["ifrs-full_ProfitLoss", "ifrs_ProfitLoss"], ["당기순이익", "당기순이익(손실)"], ("IS", "CIS")),
    "OperatingCashFlow": (["ifrs-full_CashFlowsFromUsedInOperatingActivities", "ifrs_CashFlowsFromUsedInOperatingActivities"],
                          ["영업활동현금흐름", "영업활동으로 인한 현금흐름"], ("CF",)),
    "DividendsPaid": (["ifrs-full_DividendsPaidClassifiedAsFinancingActivities", "ifrs_DividendsPaidClassifiedAsFinancingActivities"],
                      ["배당금의 지급", "배당금지급"], ("CF",)),
    "Capex": (["ifrs-full_PurchaseOfPropertyPlantAndEquipment", "ifrs_PurchaseOfPropertyPlantAndEquipment"],
              ["유형자산의 취득"], ("CF",)),
    "Assets": (["ifrs-full_Assets", "ifrs_Assets"], ["자산총계"], ("BS",)),
    "Liabilities": (["ifrs-full_Liabilities", "ifrs_Liabilities"], ["부채총계"], ("BS",)),
    "EquityParent": (["ifrs-full_EquityAttributableToOwnersOfParent", "ifrs_EquityAttributableToOwnersOfParent"],
                     ["지배기업의 소유주에게 귀속되는 자본", "지배기업 소유주지분"], ("BS",)),
    "Equity": (["ifrs-full_Equity", "ifrs_Equity"], ["자본총계"], ("BS",)),
    "CurrentAssets": (["ifrs-full_CurrentAssets", "ifrs_CurrentAssets"], ["유동자산"], ("BS",)),
    "CurrentLiabilities": (["ifrs-full_CurrentLiabilities", "ifrs_CurrentLiabilities"], ["유동부채"], ("BS",)),
    "Cash": (["ifrs-full_CashAndCashEquivalents", "ifrs_CashAndCashEquivalents"], ["현금및현금성자산"], ("BS",)),
}
FLOW_ITEMS = ("Revenue", "GrossProfit", "OperatingIncome", "NetIncomeParent", "NetIncome", "OperatingCashFlow",
              "DividendsPaid", "Capex")
# Financial firms (sales-based ratios are not meaningful). KIND uses KSIC names, where large holding companies sit in
# "기타 금융업" (KSIC 64992 지주회사), so that class is NOT treated as financial; banks, insurers, brokers (금융 지원
# 서비스업), trusts/asset managers are.
FINANCIAL_SECTORS = ("은행", "보험", "금융 지원", "신탁", "증권")


# ------------------------------------------------------------------ base panel (existing dataset)
def base_panel():
    firms = pd.read_parquet(DATA_ROOT / "daily_dataset" / "daily_firms.parquet",
                            columns=["Date", "Code", "Marcap", "Valid", "Change", "Turnover"])
    firms["Code"] = firms.Code.astype(str).str.zfill(6)
    market = pd.read_parquet(DATA_ROOT / "daily_dataset" / "daily_market.parquet",
                             columns=["Date", "IndexReturn", "IndexVol20"]).sort_values("Date")
    market = market.loc[market.Date >= firms.Date.min()].reset_index(drop=True)
    return firms, market


def next_session(dates, calendar):
    """First trading day strictly after each date (the conservative 'next trading day' rule)."""
    cal = np.asarray(calendar, dtype="datetime64[ns]")
    pos = np.searchsorted(cal, np.asarray(pd.to_datetime(dates), dtype="datetime64[ns]"), side="right")
    out = np.full(len(pos), np.datetime64("NaT"), dtype="datetime64[ns]")
    ok = pos < len(cal)
    out[ok] = cal[pos[ok]]
    return out


def read_csv(path, **kw):
    return pd.read_csv(path, encoding="utf-8-sig", **kw)


# ------------------------------------------------------------------ firm daily blocks
def derived_risk(firms, market):
    """Beta60 (Dimson: today + yesterday market), IdioVol60 (log), MAX20, Skew60 from existing returns."""
    f = firms[["Date", "Code", "Change"]].merge(market[["Date", "IndexReturn"]], on="Date", how="left")
    f["MktLag"] = f.Date.map(market.set_index("Date").IndexReturn.shift(1))
    f = f.sort_values(["Code", "Date"]).reset_index(drop=True)
    out = []
    for code, g in f.groupby("Code", sort=False):
        r, m0, m1 = g.Change.to_numpy(), g.IndexReturn.to_numpy(), g.MktLag.to_numpy()
        ok = np.isfinite(r) & np.isfinite(m0) & np.isfinite(m1)
        X = np.stack([np.ones(len(r)), np.where(ok, m0, 0), np.where(ok, m1, 0)], 1) * ok[:, None]
        y = np.where(ok, r, 0)
        cs = lambda a: np.cumsum(np.r_[np.zeros((1,) + a.shape[1:]), a], axis=0)  # noqa: E731
        xx, xy, yy, n = cs(X[:, :, None] * X[:, None, :]), cs(X * y[:, None]), cs(y * y), cs(ok.astype(float))
        idx = np.arange(1, len(r) + 1)
        lo = np.maximum(idx - 60, 0)
        A, b, s, cnt = xx[idx] - xx[lo], xy[idx] - xy[lo], yy[idx] - yy[lo], n[idx] - n[lo]
        good = cnt >= 40
        beta = np.full((len(r), 3), np.nan)
        if good.any():
            beta[good] = np.linalg.solve(A[good] + 1e-12 * np.eye(3), b[good][..., None])[..., 0]
        rss = s - np.einsum("ti,ti->t", beta, b)
        idio = np.sqrt(np.maximum(rss, 0) / np.maximum(cnt - 3, 1))
        rr = pd.Series(r)
        out.append(pd.DataFrame({
            "Date": g.Date.to_numpy(), "Code": code, "Beta60": beta[:, 1] + beta[:, 2],
            "LogIdioVol60": np.where(good, np.log(idio + 1e-8), np.nan),
            "MAX20": rr.rolling(20, min_periods=15).max().to_numpy(),
            "Skew60": rr.rolling(60, min_periods=40).skew().to_numpy()}))
    return pd.concat(out, ignore_index=True)


def investor_flows(firms, path):
    raw = read_csv(path, dtype={"Code": str})
    raw["Code"], raw["Date"] = raw.Code.str.zfill(6), pd.to_datetime(raw.Date)
    f = firms[["Date", "Code", "Marcap"]].merge(raw, on=["Date", "Code"], how="left").sort_values(["Code", "Date"])
    out = f[["Date", "Code"]].copy()
    cap = f.Marcap.where(f.Marcap > 0)
    for who in ("Foreign", "Inst", "Indiv"):
        col = f"{who}NetValue"
        if col not in f:
            continue
        net = f[col]
        out[f"{who}NetBuy"] = net / cap
        for w in (5, 20):
            out[f"{who}NetBuy{w}"] = net.groupby(f.Code).transform(lambda s: s.rolling(w, min_periods=w).sum()) / cap
    out["FlowMissing"] = f["ForeignNetValue"].isna().astype(float) if "ForeignNetValue" in f else 1.0
    return out


def foreign_ownership(firms, path):
    raw = read_csv(path, dtype={"Code": str})
    raw["Code"], raw["Date"] = raw.Code.str.zfill(6), pd.to_datetime(raw.Date)
    f = firms[["Date", "Code"]].merge(raw, on=["Date", "Code"], how="left").sort_values(["Code", "Date"])
    ratio = f.ForeignShares / f.ListedShares.where(f.ListedShares > 0)
    return pd.DataFrame({"Date": f.Date, "Code": f.Code, "ForeignRatio": ratio,
                         "ForeignExhaust": f.ForeignShares / f.ForeignLimitShares.where(f.ForeignLimitShares > 0),
                         "dForeignRatio5": ratio - ratio.groupby(f.Code).shift(5)})


def short_selling(firms, calendar, path_trade, path_balance, k200):
    f = firms[["Date", "Code", "Marcap", "Turnover"]].copy()
    value = f.Turnover * f.Marcap
    out = f[["Date", "Code"]].copy()
    banned = np.zeros(len(f), bool)
    for a, b in SHORT_BANS:
        banned |= f.Date.between(a, b).to_numpy()
    partial = f.Date.between(*SHORT_PARTIAL).to_numpy()
    if k200 is not None:
        member = f.merge(k200, on=["Date", "Code"], how="left").K200Member.fillna(0).to_numpy() > 0
        banned |= partial & ~member
    if path_trade.exists():
        raw = read_csv(path_trade, dtype={"Code": str})
        raw["Code"], raw["Date"] = raw.Code.str.zfill(6), pd.to_datetime(raw.Date)
        sv = f[["Date", "Code"]].merge(raw[["Date", "Code", "ShortValue"]], on=["Date", "Code"], how="left").ShortValue
        out["ShortRatio"] = (sv / value.where(value > 0)).where(~banned)
    if path_balance.exists():
        raw = read_csv(path_balance, dtype={"Code": str})
        raw["Code"] = raw.Code.str.zfill(6)
        raw["Date"] = next_session(raw.BaseDate, calendar)          # +1 session, then the disclosure lag
        cal = pd.DatetimeIndex(calendar)
        pos = cal.get_indexer(pd.to_datetime(raw.Date)) + SHORT_BALANCE_LAG - 1
        raw = raw.loc[(pos >= 0) & (pos < len(cal))].assign(Date=lambda d: cal[pos[(pos >= 0) & (pos < len(cal))]])
        bal = f[["Date", "Code"]].merge(raw[["Date", "Code", "ShortBalanceValue"]], on=["Date", "Code"], how="left")
        out["LogShortBalance"] = np.log1p(100 * bal.ShortBalanceValue / f.Marcap.where(f.Marcap > 0)).where(~banned)
    out["ShortBan"] = banned.astype(float)
    return out


def k200_membership(firms, path):
    raw = read_csv(path, dtype={"Code": str})
    raw["Code"] = raw.Code.str.zfill(6)
    rows = []
    for code, g in raw.sort_values("EffectiveDate").groupby("Code"):
        state = 0
        for _, r in g.iterrows():
            state = 1 if str(r.Action).strip() in ("편입", "add", "in") else 0
            rows.append({"Code": code, "From": pd.to_datetime(r.EffectiveDate), "Member": state,
                         "Announce": pd.to_datetime(r.AnnounceDate)})
    ev = pd.DataFrame(rows)
    f = firms[["Date", "Code"]].sort_values("Date")
    m = pd.merge_asof(f, ev.sort_values("From")[["From", "Code", "Member"]], left_on="Date", right_on="From",
                      by="Code", direction="backward")
    return pd.DataFrame({"Date": m.Date, "Code": m.Code, "K200Member": m.Member.fillna(0)})


# ------------------------------------------------------------------ DART financials
def parse_amount(s):
    return pd.to_numeric(s.astype(str).str.replace(",", "", regex=False).str.strip().replace({"": np.nan, "-": np.nan}),
                         errors="coerce")


def dart_quarterly(path_fin, path_filings, calendar):
    raw = read_csv(path_fin, dtype=str)
    raw["Code"] = raw.stock_code.str.zfill(6)
    raw = raw.loc[raw.reprt_code.isin(REPORTS)]
    # quarter mapping assumes a December fiscal year; firms with another fiscal month (KIND listing) are excluded
    listing = sorted((RAW / "krx").glob("kind_listing_*.csv"))
    if listing:
        kind = read_csv(listing[-1], dtype=str)
        non_dec = set(kind.loc[~kind.FiscalMonth.astype(str).str.startswith("12"), "Code"].str.zfill(6))
        raw = raw.loc[~raw.Code.isin(non_dec)]
    # consolidated first, separate only when a report has no consolidated rows
    has_cfs = raw.loc[raw.fs_div == "CFS"].groupby(["Code", "bsns_year", "reprt_code"]).size()
    key = pd.MultiIndex.from_frame(raw[["Code", "bsns_year", "reprt_code"]])
    raw = raw.loc[(raw.fs_div == "CFS") | ~key.isin(has_cfs.index)].copy()
    raw["value"], raw["cum"] = parse_amount(raw.thstrm_amount), parse_amount(raw.get("thstrm_add_amount", pd.Series(dtype=str)))
    picked, match = [], {}
    for item, (ids, names, kinds) in ACCOUNTS.items():
        sub = raw.loc[raw.sj_div.isin(kinds)]
        hit = sub.loc[sub.account_id.isin(ids)]
        miss = sub.loc[~sub.set_index(["Code", "bsns_year", "reprt_code"]).index.isin(
            hit.set_index(["Code", "bsns_year", "reprt_code"]).index) & sub.account_nm.str.strip().isin(names)]
        both = pd.concat([hit, miss]).drop_duplicates(["Code", "bsns_year", "reprt_code"], keep="first")
        match[item] = int(len(both))
        picked.append(both.assign(item=item)[["Code", "bsns_year", "reprt_code", "fs_div", "rcept_no", "item", "value", "cum"]])
    acc = pd.concat(picked, ignore_index=True)
    wide = acc.pivot_table(index=["Code", "bsns_year", "reprt_code"], columns="item", values="value", aggfunc="first")
    cumw = acc.pivot_table(index=["Code", "bsns_year", "reprt_code"], columns="item", values="cum", aggfunc="first")
    meta = acc.groupby(["Code", "bsns_year", "reprt_code"]).agg(FsDiv=("fs_div", "first"), RceptNo=("rcept_no", "first"))
    q = wide.join(meta).reset_index()
    q["year"], q["month"] = q.bsns_year.astype(int), q.reprt_code.map(REPORTS)
    q["FiscalPeriodEnd"] = pd.to_datetime(q.year.astype(str) + "-" + q.month.astype(str) + "-01") + pd.offsets.MonthEnd(0)
    q = q.sort_values(["Code", "FiscalPeriodEnd"]).reset_index(drop=True)
    # a cash-flow statement that exists but has no dividend line = no dividend paid year-to-date. December firms pay in
    # April, so the Q1 report has no line; leaving it NaN made Div_TTM missing almost everywhere (DY coverage 9%,
    # first real-data run 2026-11-20)
    if "DividendsPaid" in q and "OperatingCashFlow" in q:
        q["DividendsPaid"] = q.DividendsPaid.where(q.DividendsPaid.notna() | q.OperatingCashFlow.isna(), 0.0)
    # three-month flows (PREPROCESSING_SPEC 3.1):
    #   income statement: Q1-Q3 reports give the 3-month value; the annual report gives the year,
    #                     so Q4 = annual - (Q1 + Q2 + Q3), or annual - Q3 year-to-date when a quarter is missing;
    #   cash flow: every report gives year-to-date values, so a quarter is the difference of consecutive reports.
    cumw = cumw.reindex(pd.MultiIndex.from_frame(q[["Code", "bsns_year", "reprt_code"]]))
    grp = [q.Code, q.year]
    prev_month = q.groupby(["Code", "year"]).month.shift(1)
    for item in FLOW_ITEMS:
        if item not in q:
            q[item] = np.nan
        if ACCOUNTS[item][2] == ("CF",):
            prev = q.groupby(["Code", "year"])[item].shift(1)
            q[f"{item}_q"] = np.where(q.month == 3, q[item],
                                      np.where(prev_month == q.month - 3, q[item] - prev, np.nan))
        else:
            three = q[item].where(q.month != 12)
            n_three = three.groupby(grp).transform("count")
            sum_three = three.groupby(grp).transform("sum")
            cum = cumw[item].to_numpy() if item in cumw else np.full(len(q), np.nan)
            q3_ytd = pd.Series(np.where(q.month == 9, cum, np.nan), index=q.index).groupby(grp).transform("max")
            q4 = np.where(n_three == 3, q[item] - sum_three, q[item] - q3_ytd)
            q[f"{item}_q"] = np.where(q.month == 12, q4, three)
    # first (original) filing date: filings list, else rcept_no date
    q["RceptDt"] = pd.to_datetime(q.RceptNo.str[:8], format="%Y%m%d", errors="coerce")
    q["FirstRceptDt"], q["Corrected"] = q.RceptDt, np.nan
    if path_filings is not None and Path(path_filings).exists():
        fl = read_csv(path_filings, dtype=str)
        fl["Code"] = fl.stock_code.str.zfill(6)
        per = fl.report_nm.str.extract(r"\((\d{4})\.(\d{2})\)")
        fl["FiscalPeriodEnd"] = pd.to_datetime(per[0] + "-" + per[1] + "-01", errors="coerce") + pd.offsets.MonthEnd(0)
        fl["dt"] = pd.to_datetime(fl.rcept_dt, format="%Y%m%d", errors="coerce")
        fl["orig"] = ~fl.report_nm.str.contains("정정", na=False)
        first = fl.loc[fl.orig].groupby(["Code", "FiscalPeriodEnd"]).dt.min().rename("FirstFromList")
        q = q.merge(first, left_on=["Code", "FiscalPeriodEnd"], right_index=True, how="left")
        q["FirstRceptDt"] = q.FirstFromList.fillna(q.RceptDt)
        q["Corrected"] = (q.RceptDt > q.FirstRceptDt).astype(float)
        q = q.drop(columns="FirstFromList")
    q["AvailableFrom"] = next_session(q.FirstRceptDt, calendar)
    return q, match


def quarterly_ratios(q, financial_codes):
    q = q.sort_values(["Code", "FiscalPeriodEnd"]).reset_index(drop=True)
    g = q.groupby("Code", sort=False)
    qidx = q.year * 4 + q.month // 3
    span3 = (qidx - g.year.shift(3) * 4 - g.month.shift(3) // 3) == 3       # 4 consecutive quarters
    span4 = (qidx - g.year.shift(4) * 4 - g.month.shift(4) // 3) == 4       # same quarter one year earlier
    ttm = lambda c: g[f"{c}_q"].transform(lambda s: s.rolling(4, min_periods=4).sum()).where(span3)  # noqa: E731
    lag4 = lambda c: g[c].shift(4).where(span4)  # noqa: E731
    ni = ttm("NetIncomeParent").fillna(ttm("NetIncome"))
    eq = q.EquityParent.fillna(q.Equity)
    out = pd.DataFrame({"Code": q.Code, "FiscalPeriodEnd": q.FiscalPeriodEnd, "AvailableFrom": q.AvailableFrom})
    out["ROE"] = ni / ((eq + lag4("EquityParent").fillna(lag4("Equity"))) / 2).where(lambda s: s > 0)
    out["ROA"] = ttm("NetIncome") / ((q.Assets + lag4("Assets")) / 2)
    fin = q.Code.isin(financial_codes)
    out["OPM"] = (ttm("OperatingIncome") / ttm("Revenue").where(lambda s: s > 0)).where(~fin)
    out["GPA"] = (ttm("GrossProfit") / q.Assets).where(~fin)
    rev_q, oi_q = q["Revenue_q"], q["OperatingIncome_q"]
    out["SalesGrowth"] = rev_q / lag4("Revenue_q").where(lambda s: s > 0) - 1
    out["OIGrowth"] = (oi_q - lag4("OperatingIncome_q")) / lag4("Assets")
    out["AssetGrowth"] = q.Assets / lag4("Assets") - 1
    out["Accruals"] = (ttm("NetIncome") - ttm("OperatingCashFlow")) / ((q.Assets + lag4("Assets")) / 2)
    chg = q.NetIncome_q - lag4("NetIncome_q")
    out["SUE"] = chg / chg.groupby(q.Code).transform(lambda s: s.rolling(8, min_periods=6).std())
    out["Leverage"] = q.Liabilities / q.Assets
    out["CurrentRatio"] = (q.CurrentAssets / q.CurrentLiabilities.where(lambda s: s > 0)).where(~fin)
    out["CashToAssets"] = q.Cash / q.Assets
    out["CapexToAssets"] = ttm("Capex").abs() / q.Assets
    out["LogAssets"] = np.log(q.Assets.where(q.Assets > 0))
    out["FinancialFlag"] = fin.astype(float)
    # level inputs for the daily price-scaled ratios and market valuation
    levels = pd.DataFrame({"Code": q.Code, "AvailableFrom": q.AvailableFrom, "FiscalPeriodEnd": q.FiscalPeriodEnd,
                           "BookEquity": eq, "NI_TTM": ni, "Sales_TTM": ttm("Revenue"),
                           "CFO_TTM": ttm("OperatingCashFlow"), "Div_TTM": ttm("DividendsPaid").abs(), "Financial": fin})
    return out.replace([np.inf, -np.inf], np.nan), levels


def value_ratios(firms, levels):
    """Daily as-of B/M, E/P, S/P, CF/P, D/Y with the issuer's current market cap (Asness & Frazzini 2013)."""
    f = firms[["Date", "Code", "Marcap"]].copy()
    f["Issuer"] = f.Code.str[:5]
    issuer_cap = f.groupby(["Date", "Issuer"]).Marcap.transform("sum")
    lv = levels.dropna(subset=["AvailableFrom"]).sort_values("AvailableFrom")
    m = pd.merge_asof(f.assign(_cap=issuer_cap).sort_values("Date"), lv, left_on="Date", right_on="AvailableFrom",
                      by="Code", direction="backward")
    stale = (m.Date - m.FiscalPeriodEnd).dt.days > STALE_DAYS
    me = (m._cap / 1.0).where(m._cap > 0)
    out = pd.DataFrame({"Date": m.Date, "Code": m.Code})
    out["LogBM"] = np.log((m.BookEquity / me).where(m.BookEquity > 0))
    out["EP"] = m.NI_TTM / me
    out["EPneg"] = (m.NI_TTM < 0).astype(float).where(m.NI_TTM.notna())
    out["LogSP"] = np.log((m.Sales_TTM / me).where((m.Sales_TTM > 0) & ~m.Financial.fillna(False).astype(bool)))
    out["CFP"] = m.CFO_TTM / me
    out["DY"] = m.Div_TTM / me
    for c in ("LogBM", "EP", "EPneg", "LogSP", "CFP", "DY"):
        out.loc[stale, c] = np.nan
    out["FinMissing"] = out.EP.isna().astype(float)
    days_since = (m.Date - m.AvailableFrom).dt.days
    out["FilingDay"] = (days_since == 0).astype(float)
    out["LogDaysSinceFiling"] = np.log1p(days_since.where(~stale))
    agg = pd.DataFrame({"Date": m.Date, "me": me, "BE": m.BookEquity.where(~stale & (m.BookEquity > 0)),
                        "NI": m.NI_TTM.where(~stale), "D": m.Div_TTM.where(~stale)})
    return out, agg


# ------------------------------------------------------------------ market daily blocks
def market_block(market, firms, calendar, args, agg_val):
    m = market[["Date", "IndexVol20"]].copy().set_index("Date")
    cal = pd.DatetimeIndex(calendar)
    out = pd.DataFrame(index=cal)
    total_cap = firms.groupby("Date").Marcap.sum().reindex(cal)

    def daily(name, cols, parse_dates=("Date",)):
        p = RAW / name
        if not p.exists():
            return None
        d = read_csv(p)
        d["Date"] = pd.to_datetime(d[parse_dates[0]])
        d = d.drop_duplicates("Date", keep="last").set_index("Date").sort_index()
        return d.reindex(columns=cols).reindex(cal)          # columns missing from the file stay NaN

    r = daily("ecos/rates.csv", ["KTB3Y", "KTB10Y", "CD91", "CorpAAm3Y"])
    if r is not None:
        r = r.ffill(limit=3)
        out["RelShortRate"] = r.CD91 - r.CD91.rolling(250, min_periods=120).mean()
        out["Term"] = r.KTB10Y - r.KTB3Y
        out["Credit"] = r.CorpAAm3Y - r.KTB3Y
        out["dKTB3Y"], out["dKTB10Y"] = r.KTB3Y.diff(), r.KTB10Y.diff()
    fx = daily("ecos/fx.csv", ["USDKRW"])
    if fx is not None:
        ret = np.log(fx.USDKRW.ffill(limit=3)).diff()
        out["FXRet"], out["FXVol20"] = ret, ret.rolling(20, min_periods=15).std()
    vk = daily("krx/vkospi.csv", ["VKOSPI"])
    if vk is not None:
        out["LogVKOSPI"] = np.log(vk.VKOSPI)
        rv = 100 * np.sqrt(252) * m.IndexVol20.reindex(cal)
        out["VRP"] = (vk.VKOSPI ** 2 - rv ** 2) / 100
    mf = daily("krx/market_flows.csv", ["ForeignNet", "InstNet", "IndivNet"])
    if mf is not None:
        for c in mf:
            out[f"Mkt{c}"] = mf[c] / total_cap
            out[f"Mkt{c}20"] = mf[c].rolling(20, min_periods=20).sum() / total_cap
    br = RAW / "ecos" / "base_rate.csv"
    if br.exists():
        b = read_csv(br)
        b["Date"] = pd.to_datetime(b.MeetingDate)
        b = b.sort_values("Date")
        level = b.set_index("Date").BaseRate.reindex(cal.union(b.Date)).ffill().reindex(cal)
        out["BaseRate12m"] = level - level.shift(250)
        change = b.BaseRate.diff().fillna(0)
        last_dir = pd.Series(np.sign(change.where(change != 0)).to_numpy(), index=b.Date).ffill()
        out["RateDirection"] = last_dir.reindex(cal.union(b.Date)).ffill().reindex(cal).fillna(0)
        out["MPCDay"] = cal.isin(b.Date).astype(float)
    us = RAW / "global" / "us_daily.csv"
    audit_us = None
    if us.exists():
        u = read_csv(us)
        u["USDate"] = pd.to_datetime(u.USDate)
        u = u.sort_values("USDate").set_index("USDate")
        pos = np.searchsorted(u.index.values, cal.values, side="left") - 1      # last US date strictly before t
        used = pd.DatetimeIndex(np.where(pos >= 0, u.index.values[np.maximum(pos, 0)], np.datetime64("NaT")))
        audit_us = pd.DataFrame({"Date": cal, "USDateUsed": used})
        for c, name in (("SPX", "SPXRet"), ("NASDAQ", "NASDAQRet"), ("SOX", "SOXRet")):
            if c in u:
                lvl = np.log(u[c].to_numpy(float))[np.maximum(pos, 0)]
                lvl[pos < 0] = np.nan
                out[name] = pd.Series(lvl, index=cal).diff()
        if "VIX" in u:
            vix = u.VIX.to_numpy(float)[np.maximum(pos, 0)]
            out["LogVIX"] = np.log(vix)
            out["dVIX"] = pd.Series(vix, index=cal).diff()
    ge = RAW / "global" / "global_extra.csv"          # R23: USDate, US10Y, US2Y, WTI, DXY, Copper, Gold
    if ge.exists():
        u = read_csv(ge)
        u["USDate"] = pd.to_datetime(u.USDate)
        u = u.sort_values("USDate").set_index("USDate").ffill(limit=3)
        pos = np.searchsorted(u.index.values, cal.values, side="left") - 1
        pick = lambda c: pd.Series(np.where(pos >= 0, u[c].to_numpy(float)[np.maximum(pos, 0)], np.nan), index=cal)  # noqa: E731
        if {"US10Y", "US2Y"} <= set(u) and u.US2Y.notna().mean() > 0.9:
            out["USTerm"] = pick("US10Y") - pick("US2Y")
        elif {"US10Y", "US3M"} <= set(u):              # FRED unreachable: 10Y - 3M (Estrella & Mishkin 1998)
            out["USTerm"] = pick("US10Y") - pick("US3M")
        if "US10Y" in u:
            out["dUS10Y"] = pick("US10Y").diff()
        for c in ("WTI", "DXY", "Copper", "Gold"):
            if c in u:
                out[f"{c}Ret"] = np.log(pick(c).where(lambda s: s > 0)).diff()
    ms = daily("krx/market_short.csv", ["ShortValue", "ShortBalanceValue"])      # R16
    if ms is not None:
        out["MktShortRatio"] = ms.ShortValue / (firms.assign(v=firms.Turnover * firms.Marcap).groupby("Date").v.sum()
                                                .reindex(cal))
        bal = np.log((ms.ShortBalanceValue.shift(SHORT_BALANCE_LAG) / total_cap).where(lambda s: s > 0))
        out["MktLogShortBalance"] = bal          # detrended inside each training window would need refits; level kept
    fu = daily("krx/futures_program.csv", ["K200Spot", "K200Futures", "OpenInterest", "ForeignFuturesNet",
                                           "ProgramArbNet", "ProgramNonArbNet"])            # R21
    if fu is not None:
        out["Basis"] = (fu.K200Futures - fu.K200Spot) / fu.K200Spot
        out["dOpenInterest"] = np.log(fu.OpenInterest.where(fu.OpenInterest > 0)).diff()
        for c in ("ForeignFuturesNet", "ProgramArbNet", "ProgramNonArbNet"):
            if fu[c].notna().any():
                out[c] = fu[c] / total_cap
    dp = RAW / "kofia" / "deposits.csv"           # R22: Date, ReleaseDate, CustomerDeposits, MarginLoans
    if dp.exists():
        d = read_csv(dp)
        d["Use"] = next_session(d.ReleaseDate, calendar)
        d = d.dropna(subset=["Use"]).sort_values("Use").drop_duplicates("Use", keep="last").set_index("Use")
        for c in ("CustomerDeposits", "MarginLoans"):
            lvl = d[c].reindex(cal.union(d.index)).ffill().reindex(cal)
            out[f"{c}Growth20"] = np.log(lvl).diff(20)
    mm = RAW / "macro" / "monthly.csv"            # R24: Indicator, Period, ReleaseDate, Value (already YoY etc.)
    if mm.exists():
        d = read_csv(mm)
        d["Use"] = next_session(d.ReleaseDate, calendar)
        for name, g in d.dropna(subset=["Use"]).groupby("Indicator"):
            s = g.sort_values("Use").drop_duplicates("Use", keep="last").set_index("Use").Value
            out[f"Macro_{name}"] = s.reindex(cal.union(s.index)).ffill().reindex(cal)
    if agg_val is not None:
        # numerator and denominator over the same firms (those with the item); NaN unless they hold >= 50% of cap.
        # The first real-data run (2026-11-20) summed me over all firms: log(0) = -inf on 335 days before the first
        # filings, and DART coverage drift leaked into the level.
        total = agg_val.groupby("Date").me.sum()
        for name, col in (("MktLogBM", "BE"), ("MktLogEP", "NI"), ("MktLogDP", "D")):
            a = agg_val.loc[agg_val[col].notna() & agg_val.me.notna()].groupby("Date")
            num, den = a[col].sum(), a.me.sum()
            ratio = (num / den).where(den >= 0.5 * total.reindex(den.index))
            out[name] = np.log(ratio.where(ratio > 0)).reindex(cal)
    ban = np.zeros(len(cal))
    for a, b_ in SHORT_BANS:
        ban[(cal >= a) & (cal <= b_)] = 1
    out["ShortBanMkt"] = ban
    if args.calendar:
        out = out.join(calendar_features(cal))
    return out.reset_index().rename(columns={"index": "Date"}), audit_us


def calendar_features(cal):
    s = pd.Series(cal, index=cal)
    second_thu = {}
    for (y, mth), g in s.groupby([cal.year, cal.month]):
        target = pd.Timestamp(year=y, month=mth, day=1) + pd.offsets.WeekOfMonth(week=1, weekday=3)
        on_or_before = g[g <= target]
        second_thu[(y, mth)] = on_or_before.iloc[-1] if len(on_or_before) else g.iloc[0]
    expiry = pd.DatetimeIndex(list(second_thu.values()))
    month = pd.Series(cal.month, index=cal)
    nxt, prv = s.shift(-1), s.shift(1)
    weekday_gap = lambda a, b: np.busday_count(a.values.astype("datetime64[D]"), b.values.astype("datetime64[D]"))  # noqa: E731
    pre = np.zeros(len(cal))
    post = np.zeros(len(cal))
    ok_n, ok_p = nxt.notna().to_numpy(), prv.notna().to_numpy()
    pre[ok_n] = weekday_gap(s[ok_n] + pd.Timedelta(days=1), nxt[ok_n]) > 0
    post[ok_p] = weekday_gap(prv[ok_p] + pd.Timedelta(days=1), s[ok_p]) > 0
    quarter_end = (s.groupby([cal.year, cal.quarter]).transform("max") == s).to_numpy()
    days_after_q = (cal - (cal.to_period("Q").start_time - pd.Timedelta(days=1))).days
    feats = pd.DataFrame(index=cal)
    feats["OptionExpiry"] = cal.isin(expiry).astype(float)
    feats["QuadWitching"] = (cal.isin(expiry) & month.isin([3, 6, 9, 12]).to_numpy()).astype(float)
    feats["K200Rebalance"] = pd.Series(cal.isin(expiry) & month.isin([6, 12]).to_numpy(), index=cal).shift(1, fill_value=False).astype(float)
    feats["MonthEnd"] = (s.groupby([cal.year, cal.month]).transform("max") == s).astype(float).to_numpy()
    feats["QuarterEnd"] = quarter_end.astype(float)
    feats["PreHoliday"], feats["PostHoliday"] = pre, post
    feats["EarningsSeason"] = ((days_after_q <= 45) | ((cal.quarter == 1) & (days_after_q <= 90))).astype(float)
    for d in range(5):
        feats[f"Dow{d}"] = (cal.dayofweek == d).astype(float)
    return feats


# ------------------------------------------------------------------ relations
def relations():
    files = sorted((RAW / "krx").glob("sector_*.csv"))
    if not files:
        return None
    frames = []
    for p in files:
        d = read_csv(p, dtype={"Code": str})
        d["ValidFrom"] = pd.to_datetime(d.AsOfDate) if "AsOfDate" in d else pd.to_datetime(re.findall(r"\d{8}", p.name)[0])
        frames.append(d[["ValidFrom", "Code", "Sector"]])
    rel = pd.concat(frames, ignore_index=True)
    rel["Code"] = rel.Code.str.zfill(6)
    return rel


# ------------------------------------------------------------------ driver
def coverage(frame, firms, cols):
    valid = firms.loc[firms.Valid, ["Date", "Code"]]
    merged = valid.merge(frame, on=["Date", "Code"], how="left")
    year = merged.Date.dt.year
    return pd.DataFrame({c: merged[c].notna().groupby(year).mean() for c in cols}).T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--derived", action="store_true", help="add Beta60, LogIdioVol60, MAX20, Skew60 from prices")
    ap.add_argument("--calendar", action="store_true", help="add calendar dummies to the market block")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "audit").mkdir(exist_ok=True)
    firms, market = base_panel()
    calendar = np.sort(market.Date.unique())
    report = {"raw_dir": str(RAW), "blocks": {}}
    daily = firms[["Date", "Code"]].copy()

    def add(name, frame):
        nonlocal daily
        if frame is None:
            return
        cols = [c for c in frame.columns if c not in ("Date", "Code")]
        daily = daily.merge(frame, on=["Date", "Code"], how="left")
        report["blocks"][name] = cols

    if args.derived:
        add("risk (derived)", derived_risk(firms, market))
    p = RAW / "krx" / FLOWS_FILE
    if p.exists():
        add("investor flows", investor_flows(firms, p))
    p = RAW / "krx" / "foreign_ownership.csv"
    if p.exists():
        add("foreign ownership", foreign_ownership(firms, p))
    k200 = k200_membership(firms, RAW / "krx" / "k200_changes.csv") if (RAW / "krx" / "k200_changes.csv").exists() else None
    if k200 is not None:
        add("KOSPI200", k200)
    pt, pb = RAW / "krx" / "short_trades.csv", RAW / "krx" / "short_balance.csv"
    if pt.exists() or pb.exists():
        add("short selling", short_selling(firms, calendar, pt, pb, k200))
    agg_val = None
    rel = relations()
    pf = RAW / "dart" / "financials.csv"
    if pf.exists():
        fin_codes = set()
        if rel is not None:
            fin_codes = set(rel.loc[rel.Sector.astype(str).str.contains("|".join(FINANCIAL_SECTORS), regex=True), "Code"])
            report["financial_firms"] = len(fin_codes)
        q, match = dart_quarterly(pf, RAW / "dart" / "filings.csv", calendar)
        ratios, levels = quarterly_ratios(q, fin_codes)
        ratios.to_parquet(OUT / "firm_quarterly.parquet", index=False)
        vr, agg_val = value_ratios(firms, levels)
        add("value ratios", vr)
        report["dart"] = {"account_matches": match, "quarters": int(len(q)),
                          "corrected_share": float(q.Corrected.mean()) if q.Corrected.notna().any() else None}
        bad = q.loc[q.AvailableFrom <= q.FirstRceptDt]
        report["audit_financial_timing_violations"] = int(len(bad))
    if rel is not None:
        rel.to_parquet(OUT / "relations.parquet", index=False)
        report["relations"] = {"snapshots": sorted(rel.ValidFrom.dt.strftime("%Y-%m-%d").unique().tolist())}
    if len(daily.columns) > 2:
        daily.to_parquet(OUT / "firm_daily_extra.parquet", index=False)
        cols = [c for c in daily.columns if c not in ("Date", "Code")]
        coverage(daily, firms, cols).to_csv(OUT / "audit" / "firm_feature_coverage_by_year.csv")
    mk, audit_us = market_block(market, firms, calendar, args, agg_val)
    if len(mk.columns) > 2:
        mk.to_parquet(OUT / "market_daily_extra.parquet", index=False)
        report["blocks"]["market"] = [c for c in mk.columns if c != "Date"]
        mk.set_index("Date").notna().groupby(lambda d: d.year).mean().T.to_csv(OUT / "audit" / "market_feature_coverage_by_year.csv")
    if audit_us is not None:
        audit_us.to_csv(OUT / "audit" / "us_alignment.csv", index=False)
        report["audit_us_violations"] = int((audit_us.USDateUsed >= audit_us.Date).sum())
    (OUT / "audit" / "ingest_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str),
                                                      encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
