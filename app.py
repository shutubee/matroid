"""
Constantan Robust Matroid Studio
--------------------------------
A Streamlit prototype for robust matroid partitioning in constantan-type Cu-Ni alloys.

This app is a decision-support prototype. The property model included here is a transparent
placeholder surrogate so the workflow can be tested before connecting experimental,
CALPHAD, phase-field, or ML property models.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from itertools import product
from typing import Dict, Iterable, List, Tuple
import io
import json

import numpy as np
import pandas as pd
import streamlit as st


APP_TITLE = "Constantan Robust Matroid Studio"
APP_SUBTITLE = "Composition–Process–Property Screening for Cu–Ni Constantan-Type Alloys"


@dataclass
class TargetWindows:
    rho_min: float = 0.40
    rho_max: float = 0.62
    tcr_abs_max: float = 35.0
    hardness_min: float = 85.0
    hardness_max: float = 210.0
    strength_min: float = 280.0
    strength_max: float = 720.0
    ductility_min: float = 8.0
    stability_min: float = 55.0
    drift_max: float = 0.080


@dataclass
class LossWeights:
    rho: float = 4.0
    tcr: float = 2.0
    drift: float = 2.0
    cost: float = 0.5
    hardness: float = 1.0
    strength: float = 0.7
    ductility: float = 1.0
    stability: float = 1.0


@dataclass
class UncertaintyConfig:
    n_scenarios: int = 64
    delta_ni: float = 0.35
    delta_cold_work: float = 3.0
    delta_anneal: float = 15.0
    delta_test_temp: float = 2.0
    meas_noise_frac: float = 0.015
    pred_noise_frac: float = 0.020
    seed: int = 42


@dataclass
class MatroidCaps:
    per_ni_bucket: int = 2
    per_cold_work_class: int = 2
    per_anneal_class: int = 2
    bucket_size: int = 5
    total_selected: int = 24


BUCKETS = [
    "Electrical stability",
    "High resistivity",
    "Formability",
    "Robust manufacturing",
    "Low cost",
    "Experimental validation",
]

REQUIRED_COLUMNS = {
    "candidate_id",
    "x_Ni_pct",
    "dopant_pct",
    "cold_work_pct",
    "anneal_C",
    "test_temp_C",
    "cyclic_strain_pct",
}


def round_list(start: float, stop: float, step: float) -> List[float]:
    """Inclusive floating range helper."""
    values = []
    value = start
    while value <= stop + 1e-9:
        values.append(round(value, 4))
        value += step
    return values


def generate_candidates(
    ni_values: Iterable[float],
    dopant_values: Iterable[float],
    cold_work_values: Iterable[float],
    anneal_values: Iterable[float],
    test_temp_values: Iterable[float],
    strain_values: Iterable[float],
    max_rows: int = 5000,
) -> pd.DataFrame:
    rows = []
    for i, (ni, dop, cw, ann, tt, strain) in enumerate(
        product(ni_values, dopant_values, cold_work_values, anneal_values, test_temp_values, strain_values),
        start=1,
    ):
        if i > max_rows:
            break
        cu = 100.0 - ni - dop
        if cu <= 0:
            continue
        rows.append(
            {
                "candidate_id": f"C{i:04d}",
                "x_Cu_pct": cu,
                "x_Ni_pct": float(ni),
                "dopant_pct": float(dop),
                "cold_work_pct": float(cw),
                "anneal_C": float(ann),
                "test_temp_C": float(tt),
                "cyclic_strain_pct": float(strain),
            }
        )
    return pd.DataFrame(rows)


def validate_candidate_table(df: pd.DataFrame) -> Tuple[bool, str]:
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        return False, f"Missing required columns: {', '.join(sorted(missing))}"
    return True, "OK"


def property_surrogate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transparent placeholder surrogate.

    Units are engineering-style placeholders for screening:
    - rho_uohm_m: micro-ohm meter-ish scaled resistivity proxy
    - tcr_ppm_K: temperature coefficient proxy in ppm/K
    - hardness_HV: Vickers hardness proxy
    - strength_MPa: strength proxy
    - ductility_pct: ductility/formability proxy
    - stability_score: 0-100 robustness/microstructure proxy
    - drift_frac: cyclic resistance drift fraction proxy
    - cost_index: relative cost proxy
    """
    out = df.copy()
    ni = out["x_Ni_pct"].astype(float)
    dop = out.get("dopant_pct", 0.0).astype(float)
    cw = out["cold_work_pct"].astype(float)
    ann = out["anneal_C"].astype(float)
    tt = out["test_temp_C"].astype(float)
    strain = out["cyclic_strain_pct"].astype(float)

    # Heuristic physics-inspired proxies, intentionally simple and editable.
    out["rho_uohm_m"] = 0.22 + 0.0082 * ni + 0.010 * dop - 0.00015 * (ann - 450.0) + 0.00045 * cw
    out["tcr_ppm_K"] = 6.0 + 0.55 * np.abs(ni - 45.0) + 0.030 * np.abs(ann - 500.0) + 0.20 * np.maximum(cw - 55.0, 0)
    out["hardness_HV"] = 72.0 + 1.95 * cw - 0.060 * (ann - 450.0) + 9.0 * dop
    out["strength_MPa"] = 235.0 + 7.6 * cw - 0.13 * (ann - 450.0) + 24.0 * dop + 2.2 * ni
    out["ductility_pct"] = 52.0 - 0.58 * cw + 0.045 * (ann - 450.0) - 5.5 * dop - 0.04 * np.abs(ni - 45.0)
    out["stability_score"] = (
        96.0
        - 0.48 * np.abs(ni - 45.0)
        - 0.32 * np.abs(cw - 40.0)
        - 0.060 * np.abs(ann - 500.0)
        - 1.4 * np.maximum(strain - 0.35, 0)
    )
    out["drift_frac"] = 0.012 + 0.00072 * cw + 0.00020 * np.abs(tt - 25.0) + 0.022 * strain + 0.0025 * dop
    out["cost_index"] = 1.00 + 0.020 * ni + 0.080 * dop + 0.0008 * ann + 0.0020 * cw

    out["ductility_pct"] = out["ductility_pct"].clip(lower=0.0)
    out["stability_score"] = out["stability_score"].clip(lower=0.0, upper=100.0)
    return out


def classify_design_space(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ni_bucket"] = pd.cut(
        out["x_Ni_pct"],
        bins=[0, 38, 42, 46, 50, 100],
        labels=["Ni < 38", "38 ≤ Ni < 42", "42 ≤ Ni < 46", "46 ≤ Ni < 50", "Ni ≥ 50"],
        include_lowest=True,
    ).astype(str)
    out["cold_work_class"] = pd.cut(
        out["cold_work_pct"],
        bins=[-1, 20, 50, 1000],
        labels=["low reduction", "moderate reduction", "high reduction"],
    ).astype(str)
    out["anneal_class"] = pd.cut(
        out["anneal_C"],
        bins=[-1, 450, 600, 5000],
        labels=["low anneal", "medium anneal", "high anneal"],
    ).astype(str)
    return out


def sample_scenarios(cfg: UncertaintyConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_scenarios
    return pd.DataFrame(
        {
            "scenario": np.arange(n),
            "d_Ni": rng.uniform(-cfg.delta_ni, cfg.delta_ni, n),
            "d_cold_work": rng.uniform(-cfg.delta_cold_work, cfg.delta_cold_work, n),
            "d_anneal": rng.uniform(-cfg.delta_anneal, cfg.delta_anneal, n),
            "d_test_temp": rng.uniform(-cfg.delta_test_temp, cfg.delta_test_temp, n),
            "meas_noise": rng.normal(0.0, cfg.meas_noise_frac, n),
            "pred_noise": rng.normal(0.0, cfg.pred_noise_frac, n),
        }
    )


def apply_scenario(df: pd.DataFrame, row: pd.Series) -> pd.DataFrame:
    p = df.copy()
    p["x_Ni_pct"] = (p["x_Ni_pct"] + row["d_Ni"]).clip(0.1, 99.0)
    p["x_Cu_pct"] = 100.0 - p["x_Ni_pct"] - p["dopant_pct"]
    p["cold_work_pct"] = (p["cold_work_pct"] + row["d_cold_work"]).clip(0.0, 99.0)
    p["anneal_C"] = (p["anneal_C"] + row["d_anneal"]).clip(20.0, 1100.0)
    p["test_temp_C"] = p["test_temp_C"] + row["d_test_temp"]
    p = property_surrogate(p)

    # Global measurement/prediction uncertainty in the scenario. Using the same multiplier
    # per scenario preserves comparability and keeps the uncertainty auditable.
    multiplier = 1.0 + row["meas_noise"] + row["pred_noise"]
    for col in ["rho_uohm_m", "hardness_HV", "strength_MPa", "ductility_pct", "stability_score", "drift_frac"]:
        p[col] = p[col] * multiplier
    p["tcr_ppm_K"] = p["tcr_ppm_K"] * (1.0 + abs(row["pred_noise"]))
    return p


def margin_loss(values: pd.Series, low: float | None = None, high: float | None = None, scale: float = 1.0) -> pd.Series:
    penalty = pd.Series(np.zeros(len(values)), index=values.index, dtype=float)
    if low is not None:
        penalty += np.maximum(low - values, 0.0) / max(scale, 1e-9)
    if high is not None:
        penalty += np.maximum(values - high, 0.0) / max(scale, 1e-9)
    return penalty


def evaluate_nominal(df: pd.DataFrame, target: TargetWindows, weights: LossWeights) -> pd.DataFrame:
    out = classify_design_space(property_surrogate(df))
    rho_target = 0.5 * (target.rho_min + target.rho_max)
    rho_scale = max(target.rho_max - target.rho_min, 1e-6)

    out["rho_loss"] = np.abs(out["rho_uohm_m"] - rho_target) / rho_scale
    out["tcr_loss"] = np.maximum(np.abs(out["tcr_ppm_K"]) - target.tcr_abs_max, 0.0) / max(target.tcr_abs_max, 1e-6)
    out["drift_loss"] = np.maximum(out["drift_frac"] - target.drift_max, 0.0) / max(target.drift_max, 1e-6)
    out["hardness_loss"] = margin_loss(out["hardness_HV"], target.hardness_min, target.hardness_max, 100.0)
    out["strength_loss"] = margin_loss(out["strength_MPa"], target.strength_min, target.strength_max, 400.0)
    out["ductility_loss"] = margin_loss(out["ductility_pct"], target.ductility_min, None, 40.0)
    out["stability_loss"] = margin_loss(out["stability_score"], target.stability_min, None, 100.0)
    out["cost_loss"] = out["cost_index"] / max(out["cost_index"].median(), 1e-6)

    out["nominal_loss"] = (
        weights.rho * out["rho_loss"]
        + weights.tcr * out["tcr_loss"]
        + weights.drift * out["drift_loss"]
        + weights.hardness * out["hardness_loss"]
        + weights.strength * out["strength_loss"]
        + weights.ductility * out["ductility_loss"]
        + weights.stability * out["stability_loss"]
        + weights.cost * out["cost_loss"]
    )
    out["nominal_feasible"] = (
        (out["rho_uohm_m"].between(target.rho_min, target.rho_max))
        & (np.abs(out["tcr_ppm_K"]) <= target.tcr_abs_max)
        & (out["hardness_HV"].between(target.hardness_min, target.hardness_max))
        & (out["strength_MPa"].between(target.strength_min, target.strength_max))
        & (out["ductility_pct"] >= target.ductility_min)
        & (out["stability_score"] >= target.stability_min)
        & (out["drift_frac"] <= target.drift_max)
    )
    return out


def evaluate_robust(
    base_df: pd.DataFrame,
    target: TargetWindows,
    weights: LossWeights,
    uncertainty: UncertaintyConfig,
) -> pd.DataFrame:
    nominal = evaluate_nominal(base_df, target, weights).set_index("candidate_id", drop=False)
    scenarios = sample_scenarios(uncertainty)
    losses = []
    feas = []
    worst_rho = []
    worst_tcr = []
    worst_drift = []

    for _, scenario in scenarios.iterrows():
        scenario_df = evaluate_nominal(apply_scenario(base_df, scenario), target, weights).set_index("candidate_id")
        scenario_loss = scenario_df["nominal_loss"].rename(int(scenario["scenario"]))
        scenario_feas = scenario_df["nominal_feasible"].rename(int(scenario["scenario"]))
        losses.append(scenario_loss)
        feas.append(scenario_feas)
        worst_rho.append(scenario_df["rho_uohm_m"])
        worst_tcr.append(scenario_df["tcr_ppm_K"])
        worst_drift.append(scenario_df["drift_frac"])

    loss_table = pd.concat(losses, axis=1)
    feas_table = pd.concat(feas, axis=1)
    rho_table = pd.concat(worst_rho, axis=1)
    tcr_table = pd.concat(worst_tcr, axis=1)
    drift_table = pd.concat(worst_drift, axis=1)

    nominal["robust_loss"] = loss_table.max(axis=1)
    nominal["mean_scenario_loss"] = loss_table.mean(axis=1)
    nominal["scenario_failure_rate"] = 1.0 - feas_table.mean(axis=1)
    nominal["robust_feasible"] = feas_table.all(axis=1)
    nominal["rho_min_scenario"] = rho_table.min(axis=1)
    nominal["rho_max_scenario"] = rho_table.max(axis=1)
    nominal["tcr_max_abs_scenario"] = tcr_table.abs().max(axis=1)
    nominal["drift_max_scenario"] = drift_table.max(axis=1)

    # Conservative score: lower is better.
    nominal["priority_score"] = nominal["robust_loss"] + 8.0 * nominal["scenario_failure_rate"]
    return nominal.reset_index(drop=True)


def preferred_bucket(row: pd.Series, target: TargetWindows) -> str:
    rho_mid = 0.5 * (target.rho_min + target.rho_max)
    if row["robust_feasible"] and row["tcr_max_abs_scenario"] <= 0.65 * target.tcr_abs_max and row["drift_max_scenario"] <= 0.70 * target.drift_max:
        return "Electrical stability"
    if row["rho_uohm_m"] >= rho_mid and row["robust_feasible"]:
        return "High resistivity"
    if row["ductility_pct"] >= max(target.ductility_min + 10.0, 18.0):
        return "Formability"
    if row["stability_score"] >= target.stability_min + 15.0 and row["scenario_failure_rate"] <= 0.10:
        return "Robust manufacturing"
    if row["cost_index"] <= row.get("cost_median", row["cost_index"]):
        return "Low cost"
    return "Experimental validation"


def independent_if_added(selected: pd.DataFrame, candidate: pd.Series, caps: MatroidCaps) -> bool:
    if len(selected) >= caps.bucket_size:
        return False
    checks = [
        ("ni_bucket", caps.per_ni_bucket),
        ("cold_work_class", caps.per_cold_work_class),
        ("anneal_class", caps.per_anneal_class),
    ]
    for col, cap in checks:
        if col in selected.columns and (selected[col] == candidate[col]).sum() >= cap:
            return False
    return True


def partition_candidates(scored: pd.DataFrame, target: TargetWindows, caps: MatroidCaps) -> pd.DataFrame:
    if scored.empty:
        return scored.copy()

    df = scored.copy()
    df["cost_median"] = df["cost_index"].median()
    df["preferred_bucket"] = df.apply(lambda r: preferred_bucket(r, target), axis=1)
    df = df.sort_values(["robust_feasible", "priority_score", "robust_loss"], ascending=[False, True, True])

    assigned_ids: set[str] = set()
    partitions: Dict[str, List[pd.Series]] = {bucket: [] for bucket in BUCKETS}

    # First pass: respect preferred bucket.
    for _, row in df.iterrows():
        if len(assigned_ids) >= caps.total_selected:
            break
        cid = row["candidate_id"]
        bucket = row["preferred_bucket"]
        if cid in assigned_ids:
            continue
        selected_bucket = pd.DataFrame(partitions[bucket])
        if independent_if_added(selected_bucket, row, caps):
            candidate = row.copy()
            candidate["assigned_bucket"] = bucket
            partitions[bucket].append(candidate)
            assigned_ids.add(cid)

    # Second pass: fill underrepresented buckets with remaining candidates if independent.
    for bucket in BUCKETS:
        for _, row in df.iterrows():
            if len(assigned_ids) >= caps.total_selected or len(partitions[bucket]) >= caps.bucket_size:
                break
            cid = row["candidate_id"]
            if cid in assigned_ids:
                continue
            selected_bucket = pd.DataFrame(partitions[bucket])
            if independent_if_added(selected_bucket, row, caps):
                candidate = row.copy()
                candidate["assigned_bucket"] = bucket
                partitions[bucket].append(candidate)
                assigned_ids.add(cid)

    rows: List[pd.Series] = []
    for bucket, bucket_rows in partitions.items():
        for rank, row in enumerate(bucket_rows, start=1):
            row = row.copy()
            row["bucket_rank"] = rank
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).drop(columns=["cost_median"], errors="ignore")


def build_explanation(selected: pd.DataFrame) -> pd.DataFrame:
    if selected.empty:
        return selected
    explain = selected[[
        "candidate_id",
        "assigned_bucket",
        "bucket_rank",
        "x_Ni_pct",
        "cold_work_pct",
        "anneal_C",
        "rho_uohm_m",
        "tcr_ppm_K",
        "drift_frac",
        "hardness_HV",
        "strength_MPa",
        "ductility_pct",
        "stability_score",
        "robust_loss",
        "scenario_failure_rate",
        "ni_bucket",
        "cold_work_class",
        "anneal_class",
    ]].copy()
    explain["selection_reason"] = explain.apply(
        lambda r: (
            f"Assigned to {r['assigned_bucket']} with rank {int(r['bucket_rank'])}; "
            f"Ni bucket={r['ni_bucket']}, cold-work={r['cold_work_class']}, anneal={r['anneal_class']}; "
            f"worst-case loss={r['robust_loss']:.3f}, scenario failure={100*r['scenario_failure_rate']:.1f}%."
        ),
        axis=1,
    )
    return explain


def csv_download(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def json_download(payload: dict) -> bytes:
    return json.dumps(payload, indent=2).encode("utf-8")


def configure_sidebar() -> Tuple[TargetWindows, LossWeights, UncertaintyConfig, MatroidCaps]:
    st.sidebar.header("Target property windows")
    target = TargetWindows(
        rho_min=st.sidebar.number_input("ρ min", 0.0, 5.0, 0.40, 0.01),
        rho_max=st.sidebar.number_input("ρ max", 0.0, 5.0, 0.62, 0.01),
        tcr_abs_max=st.sidebar.number_input("|TCR| max ppm/K", 0.0, 500.0, 35.0, 1.0),
        hardness_min=st.sidebar.number_input("Hardness min HV", 0.0, 1000.0, 85.0, 5.0),
        hardness_max=st.sidebar.number_input("Hardness max HV", 0.0, 1000.0, 210.0, 5.0),
        strength_min=st.sidebar.number_input("Strength min MPa", 0.0, 2000.0, 280.0, 10.0),
        strength_max=st.sidebar.number_input("Strength max MPa", 0.0, 2000.0, 720.0, 10.0),
        ductility_min=st.sidebar.number_input("Ductility min %", 0.0, 100.0, 8.0, 1.0),
        stability_min=st.sidebar.number_input("Stability score min", 0.0, 100.0, 55.0, 1.0),
        drift_max=st.sidebar.number_input("Cyclic drift max", 0.0, 1.0, 0.080, 0.005),
    )

    with st.sidebar.expander("Robust loss weights", expanded=False):
        weights = LossWeights(
            rho=st.slider("ρ weight", 0.0, 10.0, 4.0, 0.5),
            tcr=st.slider("TCR weight", 0.0, 10.0, 2.0, 0.5),
            drift=st.slider("Drift weight", 0.0, 10.0, 2.0, 0.5),
            cost=st.slider("Cost weight", 0.0, 5.0, 0.5, 0.1),
            hardness=st.slider("Hardness weight", 0.0, 5.0, 1.0, 0.1),
            strength=st.slider("Strength weight", 0.0, 5.0, 0.7, 0.1),
            ductility=st.slider("Ductility weight", 0.0, 5.0, 1.0, 0.1),
            stability=st.slider("Stability weight", 0.0, 5.0, 1.0, 0.1),
        )

    st.sidebar.header("Uncertainty scenarios")
    uncertainty = UncertaintyConfig(
        n_scenarios=st.sidebar.slider("Scenario count", 8, 256, 64, 8),
        delta_ni=st.sidebar.slider("±Ni composition scatter", 0.0, 2.0, 0.35, 0.05),
        delta_cold_work=st.sidebar.slider("±Cold-work scatter %", 0.0, 15.0, 3.0, 0.5),
        delta_anneal=st.sidebar.slider("±Anneal drift °C", 0.0, 60.0, 15.0, 1.0),
        delta_test_temp=st.sidebar.slider("±Test-temp drift °C", 0.0, 20.0, 2.0, 0.5),
        meas_noise_frac=st.sidebar.slider("Measurement noise", 0.0, 0.10, 0.015, 0.005),
        pred_noise_frac=st.sidebar.slider("Prediction noise", 0.0, 0.15, 0.020, 0.005),
        seed=st.sidebar.number_input("Random seed", 1, 999999, 42, 1),
    )

    st.sidebar.header("Matroid capacity rules")
    caps = MatroidCaps(
        per_ni_bucket=st.sidebar.slider("Max per Ni bucket", 1, 10, 2, 1),
        per_cold_work_class=st.sidebar.slider("Max per cold-work class", 1, 10, 2, 1),
        per_anneal_class=st.sidebar.slider("Max per anneal class", 1, 10, 2, 1),
        bucket_size=st.sidebar.slider("Max designs per bucket", 1, 20, 5, 1),
        total_selected=st.sidebar.slider("Total selected designs", 1, 100, 24, 1),
    )
    return target, weights, uncertainty, caps


def load_or_generate_candidates() -> pd.DataFrame:
    st.subheader("1. Candidate ground set")
    source = st.radio("Candidate source", ["Generate design grid", "Upload CSV"], horizontal=True)

    if source == "Upload CSV":
        st.caption("Required columns: candidate_id, x_Ni_pct, dopant_pct, cold_work_pct, anneal_C, test_temp_C, cyclic_strain_pct. x_Cu_pct is optional.")
        uploaded = st.file_uploader("Upload candidate CSV", type=["csv"])
        if uploaded is None:
            st.info("Upload a CSV or switch to generated grid.")
            return pd.DataFrame()
        df = pd.read_csv(uploaded)
        ok, msg = validate_candidate_table(df)
        if not ok:
            st.error(msg)
            return pd.DataFrame()
        if "x_Cu_pct" not in df.columns:
            df["x_Cu_pct"] = 100.0 - df["x_Ni_pct"] - df["dopant_pct"]
        return df

    col1, col2, col3 = st.columns(3)
    with col1:
        ni_min = st.number_input("Ni min %", 20.0, 80.0, 36.0, 1.0)
        ni_max = st.number_input("Ni max %", 20.0, 80.0, 52.0, 1.0)
        ni_step = st.number_input("Ni step %", 0.5, 10.0, 2.0, 0.5)
    with col2:
        dopant_values_text = st.text_input("Dopant/impurity % list", "0, 0.2, 0.5")
        cold_work_text = st.text_input("Cold-work % list", "10, 30, 50, 70")
    with col3:
        anneal_text = st.text_input("Anneal °C list", "400, 500, 600")
        test_temp_text = st.text_input("Test temp °C list", "25, 80")
        strain_text = st.text_input("Cyclic strain % list", "0.1, 0.3, 0.5")

    def parse_float_list(text: str) -> List[float]:
        return [float(x.strip()) for x in text.split(",") if x.strip()]

    try:
        df = generate_candidates(
            round_list(ni_min, ni_max, ni_step),
            parse_float_list(dopant_values_text),
            parse_float_list(cold_work_text),
            parse_float_list(anneal_text),
            parse_float_list(test_temp_text),
            parse_float_list(strain_text),
        )
    except ValueError as exc:
        st.error(f"Could not parse list input: {exc}")
        return pd.DataFrame()
    st.caption(f"Generated {len(df):,} candidate alloy–process–test configurations.")
    return df


def render_results(candidates: pd.DataFrame, target: TargetWindows, weights: LossWeights, uncertainty: UncertaintyConfig, caps: MatroidCaps) -> None:
    if candidates.empty:
        return
    ok, msg = validate_candidate_table(candidates)
    if not ok:
        st.error(msg)
        return

    with st.spinner("Evaluating nominal and perturbed scenarios..."):
        scored = evaluate_robust(candidates, target, weights, uncertainty)
        selected = partition_candidates(scored, target, caps)
        explain = build_explanation(selected)

    st.subheader("2. Robust screening summary")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Candidates", f"{len(scored):,}")
    m2.metric("Nominal feasible", f"{int(scored['nominal_feasible'].sum()):,}")
    m3.metric("Robust feasible", f"{int(scored['robust_feasible'].sum()):,}")
    m4.metric("Selected", f"{len(selected):,}")

    tab1, tab2, tab3, tab4 = st.tabs(["Selected partitions", "Candidate table", "Charts", "Exports"])

    with tab1:
        st.markdown("#### Matroid-partitioned design buckets")
        if selected.empty:
            st.warning("No candidates could be selected under the current windows and capacity rules. Relax targets or capacities.")
        else:
            bucket_counts = selected.groupby("assigned_bucket").size().rename("count").reset_index()
            st.dataframe(bucket_counts, use_container_width=True, hide_index=True)
            st.dataframe(
                explain.sort_values(["assigned_bucket", "bucket_rank"]),
                use_container_width=True,
                hide_index=True,
            )

    with tab2:
        st.markdown("#### Full scored ground set")
        view_cols = [
            "candidate_id",
            "x_Cu_pct",
            "x_Ni_pct",
            "dopant_pct",
            "cold_work_pct",
            "anneal_C",
            "test_temp_C",
            "cyclic_strain_pct",
            "rho_uohm_m",
            "tcr_ppm_K",
            "drift_frac",
            "hardness_HV",
            "strength_MPa",
            "ductility_pct",
            "stability_score",
            "nominal_feasible",
            "robust_feasible",
            "scenario_failure_rate",
            "robust_loss",
            "priority_score",
            "ni_bucket",
            "cold_work_class",
            "anneal_class",
        ]
        st.dataframe(scored[view_cols].sort_values("priority_score"), use_container_width=True, hide_index=True)

    with tab3:
        st.markdown("#### Property-space view")
        chart_df = scored.copy()
        chart_df["selected"] = chart_df["candidate_id"].isin(set(selected.get("candidate_id", [])))
        st.scatter_chart(
            chart_df,
            x="x_Ni_pct",
            y="rho_uohm_m",
            color="selected",
            size="stability_score",
        )
        st.markdown("#### Robust loss by Ni bucket")
        bucket_loss = chart_df.groupby("ni_bucket", as_index=False)["robust_loss"].mean()
        st.bar_chart(bucket_loss, x="ni_bucket", y="robust_loss")
        st.markdown("#### Scenario failure rate by cold-work class")
        fail = chart_df.groupby("cold_work_class", as_index=False)["scenario_failure_rate"].mean()
        st.bar_chart(fail, x="cold_work_class", y="scenario_failure_rate")

    with tab4:
        config_payload = {
            "target_windows": asdict(target),
            "loss_weights": asdict(weights),
            "uncertainty": asdict(uncertainty),
            "matroid_caps": asdict(caps),
            "note": "Property surrogate is a transparent placeholder. Replace with validated experimental/CALPHAD/ML property models before decision use.",
        }
        c1, c2, c3 = st.columns(3)
        c1.download_button("Download scored CSV", csv_download(scored), "constantan_scored_candidates.csv", "text/csv")
        c2.download_button("Download selected CSV", csv_download(explain), "constantan_selected_partitions.csv", "text/csv")
        c3.download_button("Download config JSON", json_download(config_payload), "constantan_app_config.json", "application/json")


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption(APP_SUBTITLE)

    with st.expander("What this prototype does", expanded=True):
        st.markdown(
            """
            This app turns a constantan alloy design library into a **ground set** of
            alloy–process–test candidates, evaluates each candidate under uncertainty,
            and partitions the best candidates into interpretable design buckets.

            **Prototype workflow:** generate/upload candidates → predict properties → create
            perturbation scenarios → compute worst-case robust loss → enforce partition-matroid
            capacity rules → export selected experimental blocks.

            The built-in property equations are deliberately simple placeholder surrogates.
            Replace `property_surrogate()` with validated experimental, CALPHAD-informed,
            phase-field, Gaussian-process, or ML property models when those are available.
            """
        )

    target, weights, uncertainty, caps = configure_sidebar()
    candidates = load_or_generate_candidates()
    if not candidates.empty:
        st.dataframe(candidates.head(100), use_container_width=True, hide_index=True)
        render_results(candidates, target, weights, uncertainty, caps)


if __name__ == "__main__":
    main()
