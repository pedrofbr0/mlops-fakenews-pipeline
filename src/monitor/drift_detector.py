"""
Drift Detector — Monitor data and model drift in production.

Demonstrates:
- Population Stability Index (PSI) for data drift
- Kolmogorov-Smirnov test for distribution shifts
- Performance degradation detection
- Alerting with structured logs (extendable to Slack/PagerDuty)
- Reference vs. production data comparison
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import settings
from src.features.feature_engineering import FEATURE_COLUMNS


@dataclass
class DriftResult:
    """Result of a single feature drift check."""

    feature: str
    psi: float
    ks_statistic: float
    ks_pvalue: float
    is_drifted: bool
    severity: str  # "none", "warning", "critical"


@dataclass
class DriftReport:
    """Full drift report across all features."""

    timestamp: str
    total_features: int
    drifted_features: int
    critical_features: list[str]
    warning_features: list[str]
    results: list[DriftResult]
    overall_status: str  # "ok", "warning", "critical"


class DriftDetector:
    """Detect data drift between reference and production distributions."""

    def __init__(
        self,
        psi_warning: float = 0.1,
        psi_critical: float = 0.2,
        ks_alpha: float = 0.05,
        n_bins: int = 10,
    ):
        self.psi_warning = psi_warning
        self.psi_critical = psi_critical
        self.ks_alpha = ks_alpha
        self.n_bins = n_bins

    @staticmethod
    def _calculate_psi(reference: np.ndarray, production: np.ndarray, n_bins: int = 10) -> float:
        """
        Calculate Population Stability Index (PSI).

        PSI < 0.1  → No significant shift
        PSI 0.1-0.2 → Moderate shift (warning)
        PSI > 0.2  → Significant shift (action needed)
        """
        # Create bins from reference distribution
        eps = 1e-6
        breakpoints = np.linspace(
            min(reference.min(), production.min()) - eps,
            max(reference.max(), production.max()) + eps,
            n_bins + 1,
        )

        ref_counts = np.histogram(reference, bins=breakpoints)[0]
        prod_counts = np.histogram(production, bins=breakpoints)[0]

        # Normalize to proportions
        ref_pct = (ref_counts + eps) / (ref_counts.sum() + eps * n_bins)
        prod_pct = (prod_counts + eps) / (prod_counts.sum() + eps * n_bins)

        # PSI formula
        psi = np.sum((prod_pct - ref_pct) * np.log(prod_pct / ref_pct))
        return float(psi)

    def check_feature_drift(
        self, feature: str, reference: np.ndarray, production: np.ndarray
    ) -> DriftResult:
        """Check drift for a single feature using PSI and KS-test."""
        # Remove NaN/Inf
        ref_clean = reference[np.isfinite(reference)]
        prod_clean = production[np.isfinite(production)]

        if len(ref_clean) < 10 or len(prod_clean) < 10:
            return DriftResult(
                feature=feature,
                psi=0.0,
                ks_statistic=0.0,
                ks_pvalue=1.0,
                is_drifted=False,
                severity="none",
            )

        # PSI
        psi = self._calculate_psi(ref_clean, prod_clean, self.n_bins)

        # KS-test
        ks_stat, ks_pvalue = stats.ks_2samp(ref_clean, prod_clean)

        # Determine severity
        if psi >= self.psi_critical or ks_pvalue < self.ks_alpha / 10:
            severity = "critical"
        elif psi >= self.psi_warning or ks_pvalue < self.ks_alpha:
            severity = "warning"
        else:
            severity = "none"

        is_drifted = severity != "none"

        return DriftResult(
            feature=feature,
            psi=round(psi, 6),
            ks_statistic=round(ks_stat, 6),
            ks_pvalue=round(ks_pvalue, 6),
            is_drifted=is_drifted,
            severity=severity,
        )

    def check_all_features(
        self,
        reference_df: pd.DataFrame,
        production_df: pd.DataFrame,
        feature_cols: Optional[list[str]] = None,
    ) -> DriftReport:
        """Run drift checks across all features."""
        feature_cols = feature_cols or [
            c for c in FEATURE_COLUMNS if c in reference_df.columns and c in production_df.columns
        ]

        results = []
        for col in feature_cols:
            result = self.check_feature_drift(
                feature=col,
                reference=reference_df[col].values,
                production=production_df[col].values,
            )
            results.append(result)

        critical = [r.feature for r in results if r.severity == "critical"]
        warning = [r.feature for r in results if r.severity == "warning"]
        drifted = len(critical) + len(warning)

        if critical:
            overall = "critical"
        elif warning:
            overall = "warning"
        else:
            overall = "ok"

        report = DriftReport(
            timestamp=pd.Timestamp.now().isoformat(),
            total_features=len(feature_cols),
            drifted_features=drifted,
            critical_features=critical,
            warning_features=warning,
            results=results,
            overall_status=overall,
        )

        self._log_report(report)
        return report

    def _log_report(self, report: DriftReport) -> None:
        """Log drift report with structured output."""
        logger.info(f"{'='*60}")
        logger.info(f"DRIFT REPORT — {report.timestamp}")
        logger.info(f"Status: {report.overall_status.upper()}")
        logger.info(f"Features checked: {report.total_features}")
        logger.info(f"Features drifted: {report.drifted_features}")

        if report.critical_features:
            logger.warning(f"CRITICAL drift: {', '.join(report.critical_features)}")
        if report.warning_features:
            logger.warning(f"WARNING drift: {', '.join(report.warning_features)}")

        logger.info(f"\n{'Feature':<30} {'PSI':>8} {'KS-stat':>8} {'Status':>10}")
        logger.info("-" * 60)
        for r in report.results:
            status_icon = {"none": "✓", "warning": "⚠", "critical": "✗"}[r.severity]
            logger.info(
                f"{r.feature:<30} {r.psi:>8.4f} {r.ks_statistic:>8.4f} "
                f"{status_icon:>10}"
            )
        logger.info(f"{'='*60}")

    def should_retrain(self, report: DriftReport, max_critical: int = 2) -> bool:
        """Decide if model should be retrained based on drift."""
        if len(report.critical_features) >= max_critical:
            logger.warning("RETRAIN RECOMMENDED: Multiple critical drifts detected.")
            return True
        return False


class PerformanceMonitor:
    """Monitor model prediction quality over time."""

    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.predictions: list[dict] = []

    def log_prediction(
        self, predicted: int, probability: float, actual: Optional[int] = None
    ) -> None:
        """Log a single prediction for monitoring."""
        self.predictions.append({
            "predicted": predicted,
            "probability": probability,
            "actual": actual,
            "timestamp": pd.Timestamp.now(),
        })

        # Keep window bounded
        if len(self.predictions) > self.window_size * 2:
            self.predictions = self.predictions[-self.window_size:]

    def get_prediction_stats(self) -> dict:
        """Compute prediction distribution statistics."""
        if not self.predictions:
            return {}

        recent = self.predictions[-self.window_size:]
        probs = [p["probability"] for p in recent]

        return {
            "count": len(recent),
            "mean_probability": round(np.mean(probs), 4),
            "std_probability": round(np.std(probs), 4),
            "pct_positive": round(
                sum(1 for p in recent if p["predicted"] == 1) / len(recent), 4
            ),
            "pct_high_confidence": round(
                sum(1 for p in probs if p > 0.85 or p < 0.15) / len(probs), 4
            ),
        }


if __name__ == "__main__":
    # Demo with synthetic data
    np.random.seed(42)
    n = 1000

    reference = pd.DataFrame({
        col: np.random.randn(n) for col in FEATURE_COLUMNS
    })
    # Simulate drift in some features
    production = reference.copy()
    production["cascade_size"] += np.random.randn(n) * 2  # Add drift
    production["spread_speed_mean"] *= 1.5  # Scale shift

    detector = DriftDetector()
    report = detector.check_all_features(reference, production)

    if detector.should_retrain(report):
        logger.info("→ Triggering retraining pipeline...")
