"""
Deterministic analytics engine using pandas.
All computations here are rule-based, auditable, and reproducible.
The AI layer (agents) interprets these results — it never recomputes them.
"""

import pandas as pd
import numpy as np
from datetime import datetime, date
from typing import Optional
import warnings

warnings.filterwarnings("ignore")


# ─── Data Validation ──────────────────────────────────────────────────────────

def _auto_map_columns(df: pd.DataFrame, quality_report: dict) -> pd.DataFrame:
    """
    Intelligently map arbitrary column names to the expected schema.
    Handles CSVs from various sources (ServiceNow exports, custom spreadsheets, etc.)
    """
    cols = list(df.columns)
    mapped = {}

    def find_col(candidates, existing_name=None):
        """Find the first column matching any candidate pattern."""
        if existing_name and existing_name in cols:
            return existing_name
        for pattern in candidates:
            for col in cols:
                if pattern in col:
                    return col
        return None

    # vendor → vendor, supplier, provider, company, tool, application
    vendor_col = find_col(["vendor", "supplier", "provider", "company", "application", "tool", "product"], "vendor")
    if vendor_col and vendor_col != "vendor":
        mapped[vendor_col] = "vendor"
        quality_report["warnings"].append(f"Mapped '{vendor_col}' → 'vendor'")

    # service_name → service, name, item, description
    svc_col = find_col(["service_name", "service", "item_name", "item", "description", "name"], "service_name")
    if svc_col and svc_col != "service_name":
        mapped[svc_col] = "service_name"
        quality_report["warnings"].append(f"Mapped '{svc_col}' → 'service_name'")

    # department → department, dept, team, owner, group, business_area
    dept_col = find_col(["department", "dept", "team", "owner", "group", "business_area", "division"], "department")
    if dept_col and dept_col != "department":
        mapped[dept_col] = "department"
        quality_report["warnings"].append(f"Mapped '{dept_col}' → 'department'")

    # cost_category → category, type, cost_type, spend_category, class
    cat_col = find_col(["cost_category", "category", "cost_type", "spend_category", "spend_type_detail", "class"], "cost_category")
    if cat_col and cat_col != "cost_category":
        mapped[cat_col] = "cost_category"
        quality_report["warnings"].append(f"Mapped '{cat_col}' → 'cost_category'")

    # annual_cost → annual_cost, annual_spend, yearly_cost, total_cost, amount, spend, cost, price
    cost_col = find_col(["annual_cost", "annual_spend", "yearly_cost", "total_annual", "total_cost",
                          "amount", "annual_amount", "cost", "spend", "price", "budget"], "annual_cost")
    if cost_col and cost_col != "annual_cost":
        mapped[cost_col] = "annual_cost"
        quality_report["warnings"].append(f"Mapped '{cost_col}' → 'annual_cost'")

    # monthly_cost
    mo_col = find_col(["monthly_cost", "monthly_spend", "monthly_amount", "mo_cost"], "monthly_cost")
    if mo_col and mo_col != "monthly_cost":
        mapped[mo_col] = "monthly_cost"

    # utilization_pct → utilization, util, usage, utilisation
    util_col = find_col(["utilization_pct", "utilization", "utilisation", "util_pct", "usage_pct", "usage"], "utilization_pct")
    if util_col and util_col != "utilization_pct":
        mapped[util_col] = "utilization_pct"
        quality_report["warnings"].append(f"Mapped '{util_col}' → 'utilization_pct'")

    # contract_end_date → end_date, expiry, expiration, renewal_date
    end_col = find_col(["contract_end_date", "end_date", "expiry_date", "expiration_date", "renewal_date", "expires"], "contract_end_date")
    if end_col and end_col != "contract_end_date":
        mapped[end_col] = "contract_end_date"

    # contract_start_date
    start_col = find_col(["contract_start_date", "start_date", "inception_date", "effective_date"], "contract_start_date")
    if start_col and start_col != "contract_start_date":
        mapped[start_col] = "contract_start_date"

    # headcount_supported → headcount, users, seats, licenses, employees
    hc_col = find_col(["headcount_supported", "headcount", "users", "seats", "licenses", "employees", "user_count"], "headcount_supported")
    if hc_col and hc_col != "headcount_supported":
        mapped[hc_col] = "headcount_supported"

    # spend_type → spend_type, capex_opex, type
    st_col = find_col(["spend_type", "capex_opex", "financial_type"], "spend_type")
    if st_col and st_col != "spend_type":
        mapped[st_col] = "spend_type"

    if mapped:
        df = df.rename(columns=mapped)

    return df


def validate_and_clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Clean and validate the uploaded DataFrame.
    Handles arbitrary CSV formats by auto-detecting and mapping column names.
    Returns the cleaned DataFrame and a quality report dict.
    """
    quality_report = {
        "original_rows": len(df),
        "original_cols": len(df.columns),
        "issues": [],
        "warnings": [],
    }

    df = df.copy()

    # Normalize column names: lowercase, strip whitespace, replace spaces with underscores
    df.columns = [c.strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns]

    # Auto-map columns from arbitrary naming to expected schema
    df = _auto_map_columns(df, quality_report)

    # Drop completely empty rows
    initial_len = len(df)
    df.dropna(how="all", inplace=True)
    dropped = initial_len - len(df)
    if dropped > 0:
        quality_report["issues"].append(f"Dropped {dropped} completely empty rows")

    # Coerce numeric columns
    for col in ["annual_cost", "monthly_cost", "utilization_pct", "headcount_supported"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(r"[$,%,]", "", regex=True), errors="coerce")

    # If monthly_cost exists but annual_cost is missing, derive it
    if "monthly_cost" in df.columns and "annual_cost" in df.columns:
        mask = df["annual_cost"].isna() & df["monthly_cost"].notna()
        df.loc[mask, "annual_cost"] = df.loc[mask, "monthly_cost"] * 12

    # If only monthly_cost exists, create annual_cost
    if "monthly_cost" in df.columns and "annual_cost" not in df.columns:
        df["annual_cost"] = df["monthly_cost"] * 12
        quality_report["warnings"].append("Derived 'annual_cost' from monthly_cost * 12")

    # If only annual_cost exists, derive monthly
    if "annual_cost" in df.columns and "monthly_cost" not in df.columns:
        df["monthly_cost"] = df["annual_cost"] / 12

    # Synthesize missing required columns with sensible defaults
    if "vendor" not in df.columns and "service_name" in df.columns:
        df["vendor"] = df["service_name"]
        quality_report["warnings"].append("No vendor column found — using service_name as vendor")
    elif "vendor" not in df.columns:
        df["vendor"] = "Unknown Vendor"
        quality_report["warnings"].append("No vendor column found — defaulting to 'Unknown Vendor'")

    if "service_name" not in df.columns and "vendor" in df.columns:
        df["service_name"] = df["vendor"]

    if "department" not in df.columns:
        df["department"] = "IT Operations"
        quality_report["warnings"].append("No department column found — defaulting to 'IT Operations'")

    if "cost_category" not in df.columns:
        # Try to infer from spend_type if it exists
        if "spend_type" in df.columns:
            df["cost_category"] = df["spend_type"]
        else:
            df["cost_category"] = "Other"
        quality_report["warnings"].append("No cost_category column found — defaulting to 'Other'")

    # Clamp utilization to 0-100
    if "utilization_pct" in df.columns:
        bad_util = df["utilization_pct"] > 100
        if bad_util.any():
            quality_report["warnings"].append(f"{bad_util.sum()} rows have utilization > 100% (clamped to 100)")
            df.loc[bad_util, "utilization_pct"] = 100.0

    # Parse date columns
    for col in ["contract_start_date", "contract_end_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    quality_report["clean_rows"] = len(df)
    quality_report["null_counts"] = df.isnull().sum().to_dict()

    return df, quality_report


def compute_data_quality_score(df: pd.DataFrame, quality_report: dict) -> dict:
    """
    Compute a 0-100 data quality score based on completeness and consistency.
    """
    from src.models.schemas import REQUIRED_COLUMNS, OPTIONAL_COLUMNS

    score = 100
    details = {}

    # Required column presence (40 pts)
    present_required = [c for c in REQUIRED_COLUMNS if c in df.columns]
    required_score = (len(present_required) / len(REQUIRED_COLUMNS)) * 40
    score = score - (40 - required_score)
    details["required_columns_present"] = f"{len(present_required)}/{len(REQUIRED_COLUMNS)}"

    # Optional column presence (20 pts)
    present_optional = [c for c in OPTIONAL_COLUMNS if c in df.columns]
    optional_score = (len(present_optional) / len(OPTIONAL_COLUMNS)) * 20
    score = score - (20 - optional_score)
    details["optional_columns_present"] = f"{len(present_optional)}/{len(OPTIONAL_COLUMNS)}"

    # Null rate in required columns (30 pts)
    if present_required:
        null_pct = df[present_required].isnull().mean().mean()
        completeness_score = (1 - null_pct) * 30
        score = score - (30 - completeness_score)
        details["required_field_completeness"] = f"{(1-null_pct)*100:.1f}%"

    # Annual cost validity (10 pts)
    if "annual_cost" in df.columns:
        valid_cost = df["annual_cost"].notna() & (df["annual_cost"] > 0)
        cost_score = (valid_cost.mean()) * 10
        score = score - (10 - cost_score)
        details["valid_cost_rows"] = f"{valid_cost.sum()}/{len(df)}"

    return {
        "score": max(0, round(score)),
        "grade": _score_to_grade(score),
        "details": details,
        "issues": quality_report.get("issues", []),
        "warnings": quality_report.get("warnings", []),
    }


def _score_to_grade(score: float) -> str:
    if score >= 90: return "A (Excellent)"
    if score >= 80: return "B (Good)"
    if score >= 70: return "C (Acceptable)"
    if score >= 60: return "D (Poor)"
    return "F (Critical Issues)"


# ─── Spend Aggregations ───────────────────────────────────────────────────────

def compute_spend_summary(df: pd.DataFrame) -> dict:
    """High-level spend summary for the executive dashboard."""
    if "annual_cost" not in df.columns:
        return {}

    valid = df[df["annual_cost"].notna() & (df["annual_cost"] > 0)]
    total = valid["annual_cost"].sum()

    summary = {
        "total_annual_spend": total,
        "total_monthly_spend": total / 12,
        "record_count": len(valid),
        "unique_vendors": valid["vendor"].nunique() if "vendor" in valid.columns else 0,
        "unique_departments": valid["department"].nunique() if "department" in valid.columns else 0,
        "avg_spend_per_vendor": total / valid["vendor"].nunique() if "vendor" in valid.columns and valid["vendor"].nunique() > 0 else 0,
        "top_vendor_concentration": _top_n_concentration(valid, "vendor", 3),
    }

    if "utilization_pct" in valid.columns:
        util = valid["utilization_pct"].dropna()
        summary["avg_utilization"] = round(util.mean(), 1)
        summary["pct_underutilized"] = round((util < 50).mean() * 100, 1)

    if "spend_type" in valid.columns:
        capex = valid[valid["spend_type"].str.upper() == "CAPEX"]["annual_cost"].sum()
        opex = valid[valid["spend_type"].str.upper() == "OPEX"]["annual_cost"].sum()
        summary["capex_spend"] = capex
        summary["opex_spend"] = opex
        summary["capex_pct"] = round(capex / total * 100, 1) if total > 0 else 0

    return summary


def _top_n_concentration(df: pd.DataFrame, col: str, n: int) -> float:
    """Return the % of total spend concentrated in top N items."""
    if col not in df.columns or "annual_cost" not in df.columns:
        return 0.0
    total = df["annual_cost"].sum()
    if total == 0:
        return 0.0
    top_n = df.groupby(col)["annual_cost"].sum().nlargest(n).sum()
    return round(top_n / total * 100, 1)


def compute_spend_by_category(df: pd.DataFrame) -> pd.DataFrame:
    """Break down spend by cost_category."""
    if "cost_category" not in df.columns or "annual_cost" not in df.columns:
        return pd.DataFrame()

    result = (
        df[df["annual_cost"].notna()]
        .groupby("cost_category")
        .agg(
            total_spend=("annual_cost", "sum"),
            vendor_count=("vendor", "nunique"),
            record_count=("annual_cost", "count"),
        )
        .reset_index()
        .sort_values("total_spend", ascending=False)
    )
    total = result["total_spend"].sum()
    result["pct_of_total"] = (result["total_spend"] / total * 100).round(1)
    return result


def compute_spend_by_department(df: pd.DataFrame) -> pd.DataFrame:
    """Break down spend by department."""
    if "department" not in df.columns or "annual_cost" not in df.columns:
        return pd.DataFrame()

    result = (
        df[df["annual_cost"].notna()]
        .groupby("department")
        .agg(
            total_spend=("annual_cost", "sum"),
            service_count=("service_name", "nunique") if "service_name" in df.columns else ("annual_cost", "count"),
            avg_utilization=("utilization_pct", "mean") if "utilization_pct" in df.columns else ("annual_cost", lambda x: np.nan),
        )
        .reset_index()
        .sort_values("total_spend", ascending=False)
    )
    total = result["total_spend"].sum()
    result["pct_of_total"] = (result["total_spend"] / total * 100).round(1)
    if "avg_utilization" in result.columns:
        result["avg_utilization"] = result["avg_utilization"].round(1)
    return result


def compute_top_vendors(df: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    """Return top N vendors by annual spend."""
    if "vendor" not in df.columns or "annual_cost" not in df.columns:
        return pd.DataFrame()

    result = (
        df[df["annual_cost"].notna()]
        .groupby("vendor")
        .agg(
            total_spend=("annual_cost", "sum"),
            service_count=("service_name", "nunique") if "service_name" in df.columns else ("annual_cost", "count"),
            department_count=("department", "nunique") if "department" in df.columns else ("annual_cost", "count"),
            avg_utilization=("utilization_pct", "mean") if "utilization_pct" in df.columns else ("annual_cost", lambda x: np.nan),
        )
        .reset_index()
        .sort_values("total_spend", ascending=False)
        .head(n)
    )
    total = df["annual_cost"].sum()
    result["pct_of_total"] = (result["total_spend"] / total * 100).round(1)
    if "avg_utilization" in result.columns:
        result["avg_utilization"] = result["avg_utilization"].round(1)
    return result


# ─── Efficiency & Risk Analysis ───────────────────────────────────────────────

def detect_underutilized_services(df: pd.DataFrame, threshold: float = 50.0) -> pd.DataFrame:
    """
    Flag services with utilization below the threshold.
    Low utilization is a primary indicator of waste.
    """
    if "utilization_pct" not in df.columns or "annual_cost" not in df.columns:
        return pd.DataFrame()

    mask = df["utilization_pct"].notna() & (df["utilization_pct"] < threshold)
    result = df[mask].copy()

    if result.empty:
        return pd.DataFrame()

    cols = [c for c in ["vendor", "service_name", "department", "cost_category",
                         "annual_cost", "utilization_pct", "contract_type", "notes"] if c in result.columns]
    result = result[cols].sort_values("annual_cost", ascending=False)

    # Estimate waste: proportional to unused capacity
    result["waste_estimate"] = (result["annual_cost"] * (1 - result["utilization_pct"] / 100)).round(0)
    return result.reset_index(drop=True)


def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Statistical anomaly detection using IQR method.
    Returns records where annual_cost is an outlier within their cost_category.
    """
    if "annual_cost" not in df.columns or "cost_category" not in df.columns:
        return pd.DataFrame()

    anomalies = []
    for category, group in df.groupby("cost_category"):
        if len(group) < 3:
            continue
        q1 = group["annual_cost"].quantile(0.25)
        q3 = group["annual_cost"].quantile(0.75)
        iqr = q3 - q1
        upper = q3 + 2.0 * iqr
        outliers = group[group["annual_cost"] > upper].copy()
        if not outliers.empty:
            outliers["anomaly_reason"] = f"Spend > 2x IQR upper bound (${upper:,.0f}) for {category}"
            anomalies.append(outliers)

    if not anomalies:
        return pd.DataFrame()

    result = pd.concat(anomalies)
    cols = [c for c in ["vendor", "service_name", "department", "cost_category",
                         "annual_cost", "anomaly_reason"] if c in result.columns]
    return result[cols].sort_values("annual_cost", ascending=False).reset_index(drop=True)


def get_renewal_alerts(df: pd.DataFrame, days_ahead: int = 180) -> pd.DataFrame:
    """
    Identify contracts expiring within `days_ahead` days.
    Critical for avoiding auto-renewals and enabling renegotiation.
    """
    if "contract_end_date" not in df.columns:
        return pd.DataFrame()

    today = pd.Timestamp.now()
    cutoff = today + pd.Timedelta(days=days_ahead)

    mask = (
        df["contract_end_date"].notna()
        & (df["contract_end_date"] >= today)
        & (df["contract_end_date"] <= cutoff)
    )
    result = df[mask].copy()

    if result.empty:
        return pd.DataFrame()

    result["days_until_renewal"] = (result["contract_end_date"] - today).dt.days.astype(int)

    urgency_bins = [0, 30, 90, 180]
    urgency_labels = ["URGENT (<30 days)", "HIGH (30-90 days)", "MEDIUM (90-180 days)"]
    result["urgency"] = pd.cut(result["days_until_renewal"], bins=urgency_bins, labels=urgency_labels)

    cols = [c for c in ["vendor", "service_name", "department", "annual_cost",
                         "contract_type", "contract_end_date", "days_until_renewal", "urgency"] if c in result.columns]
    return result[cols].sort_values("days_until_renewal").reset_index(drop=True)


def detect_duplicate_tools(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify potential duplicate tools in the same cost_category and department.
    Duplicates represent consolidation opportunities.
    """
    if "cost_category" not in df.columns or "department" not in df.columns:
        return pd.DataFrame()

    duplicates = []
    for (dept, cat), group in df.groupby(["department", "cost_category"]):
        if len(group) > 1:
            row = {
                "department": dept,
                "cost_category": cat,
                "tool_count": len(group),
                "vendors": " | ".join(group["vendor"].unique()) if "vendor" in group.columns else "",
                "combined_spend": group["annual_cost"].sum() if "annual_cost" in group.columns else 0,
                "services": " | ".join(group["service_name"].unique()) if "service_name" in group.columns else "",
            }
            duplicates.append(row)

    if not duplicates:
        return pd.DataFrame()

    result = pd.DataFrame(duplicates).sort_values("combined_spend", ascending=False)
    return result[result["tool_count"] > 1].reset_index(drop=True)


def compute_cost_per_user(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute annual cost per supported headcount for services with headcount data.
    High cost-per-user indicates pricing inefficiency or over-provisioning.
    """
    if "headcount_supported" not in df.columns or "annual_cost" not in df.columns:
        return pd.DataFrame()

    mask = df["headcount_supported"].notna() & (df["headcount_supported"] > 0) & df["annual_cost"].notna()
    result = df[mask].copy()
    result["cost_per_user"] = (result["annual_cost"] / result["headcount_supported"]).round(2)

    cols = [c for c in ["vendor", "service_name", "department", "annual_cost",
                         "headcount_supported", "cost_per_user", "utilization_pct"] if c in result.columns]
    return result[cols].sort_values("cost_per_user", ascending=False).reset_index(drop=True)


# ─── TBM Mapping ──────────────────────────────────────────────────────────────

def compute_tbm_spend_pools(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map IT spend into TBM cost pools (Infrastructure, Applications, Management).
    """
    from src.models.schemas import TBM_COST_POOL_MAP, TBM_VALUE_STREAM_MAP

    if "cost_category" not in df.columns or "annual_cost" not in df.columns:
        return pd.DataFrame()

    result = df.copy()
    result["tbm_cost_pool"] = result["cost_category"].map(TBM_COST_POOL_MAP).fillna("IT Applications")

    if "department" in result.columns:
        result["tbm_value_stream"] = result["department"].map(TBM_VALUE_STREAM_MAP).fillna("Operational Efficiency")

    pool_summary = (
        result[result["annual_cost"].notna()]
        .groupby("tbm_cost_pool")
        .agg(
            total_spend=("annual_cost", "sum"),
            category_count=("cost_category", "nunique"),
            vendor_count=("vendor", "nunique") if "vendor" in result.columns else ("annual_cost", "count"),
        )
        .reset_index()
    )
    total = pool_summary["total_spend"].sum()
    pool_summary["pct_of_total"] = (pool_summary["total_spend"] / total * 100).round(1)
    return pool_summary.sort_values("total_spend", ascending=False)


def compute_value_stream_spend(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate spend by TBM value stream."""
    from src.models.schemas import TBM_VALUE_STREAM_MAP

    if "department" not in df.columns or "annual_cost" not in df.columns:
        return pd.DataFrame()

    result = df.copy()
    result["value_stream"] = result["department"].map(TBM_VALUE_STREAM_MAP).fillna("Operational Efficiency")

    vs_summary = (
        result[result["annual_cost"].notna()]
        .groupby("value_stream")
        .agg(
            total_spend=("annual_cost", "sum"),
            service_count=("service_name", "nunique") if "service_name" in result.columns else ("annual_cost", "count"),
            department_count=("department", "nunique"),
        )
        .reset_index()
    )
    total = vs_summary["total_spend"].sum()
    vs_summary["pct_of_total"] = (vs_summary["total_spend"] / total * 100).round(1)
    return vs_summary.sort_values("total_spend", ascending=False)


# ─── Full Analytics Bundle ────────────────────────────────────────────────────

def compute_full_analytics(df: pd.DataFrame) -> dict:
    """
    Compute the complete analytics bundle in one call.
    This is passed to all agents as their primary input.
    Returns a dict of DataFrames and scalar metrics.
    """
    summary = compute_spend_summary(df)
    by_category = compute_spend_by_category(df)
    by_department = compute_spend_by_department(df)
    top_vendors = compute_top_vendors(df, n=15)
    underutilized = detect_underutilized_services(df)
    anomalies = detect_anomalies(df)
    renewals = get_renewal_alerts(df)
    duplicates = detect_duplicate_tools(df)
    cost_per_user = compute_cost_per_user(df)
    tbm_pools = compute_tbm_spend_pools(df)
    value_streams = compute_value_stream_spend(df)

    # Compute total waste estimate
    waste_total = underutilized["waste_estimate"].sum() if not underutilized.empty and "waste_estimate" in underutilized.columns else 0

    return {
        "summary": summary,
        "by_category": by_category,
        "by_department": by_department,
        "top_vendors": top_vendors,
        "underutilized": underutilized,
        "anomalies": anomalies,
        "renewals": renewals,
        "duplicates": duplicates,
        "cost_per_user": cost_per_user,
        "tbm_pools": tbm_pools,
        "value_streams": value_streams,
        "waste_estimate_total": waste_total,
    }


def format_analytics_for_llm(analytics: dict) -> str:
    """
    Serialize the analytics bundle into a concise text summary for LLM consumption.
    Keeps token usage efficient while preserving all critical facts.
    """
    def fmt_usd(v):
        if v >= 1_000_000:
            return f"${v/1_000_000:.2f}M"
        if v >= 1_000:
            return f"${v/1_000:.0f}K"
        return f"${v:.0f}"

    def df_to_text(df: pd.DataFrame, max_rows: int = 15) -> str:
        if df is None or df.empty:
            return "  (no data)"
        return df.head(max_rows).to_string(index=False)

    lines = ["=== ENTERPRISE IT SPEND ANALYTICS SUMMARY ===\n"]

    s = analytics.get("summary", {})
    if s:
        lines.append("--- SPEND OVERVIEW ---")
        lines.append(f"Total Annual IT Spend: {fmt_usd(s.get('total_annual_spend', 0))}")
        lines.append(f"Total Monthly Spend: {fmt_usd(s.get('total_monthly_spend', 0))}")
        lines.append(f"Active Services: {s.get('record_count', 0)} | Unique Vendors: {s.get('unique_vendors', 0)} | Departments: {s.get('unique_departments', 0)}")
        if "avg_utilization" in s:
            lines.append(f"Average Utilization: {s.get('avg_utilization', 0)}% | Underutilized Services (<50%): {s.get('pct_underutilized', 0)}%")
        if "capex_spend" in s:
            lines.append(f"CAPEX: {fmt_usd(s.get('capex_spend', 0))} ({s.get('capex_pct', 0)}%) | OPEX: {fmt_usd(s.get('opex_spend', 0))}")
        if "top_vendor_concentration" in s:
            lines.append(f"Top-3 Vendor Spend Concentration: {s.get('top_vendor_concentration', 0)}%")
        waste = analytics.get("waste_estimate_total", 0)
        if waste > 0:
            lines.append(f"Estimated Waste (underutilized): {fmt_usd(waste)}")
        lines.append("")

    for section, title in [
        ("by_category", "SPEND BY CATEGORY"),
        ("by_department", "SPEND BY DEPARTMENT"),
        ("top_vendors", "TOP VENDORS"),
        ("tbm_pools", "TBM COST POOLS"),
        ("value_streams", "VALUE STREAM SPEND"),
    ]:
        df = analytics.get(section)
        if df is not None and not df.empty:
            lines.append(f"--- {title} ---")
            lines.append(df_to_text(df))
            lines.append("")

    for section, title in [
        ("underutilized", "UNDERUTILIZED SERVICES (< 50% utilization)"),
        ("anomalies", "SPEND ANOMALIES"),
        ("renewals", "CONTRACT RENEWALS (next 180 days)"),
        ("duplicates", "POTENTIAL DUPLICATE TOOLS"),
    ]:
        df = analytics.get(section)
        if df is not None and not df.empty:
            lines.append(f"--- {title} ---")
            lines.append(df_to_text(df))
            lines.append("")

    return "\n".join(lines)
