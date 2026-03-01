"""Pairwise correlation balance validator for synthetic data generation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import logging
import math

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from ..core.database import DatabaseManager
from ..core.config import ConfigManager


class CorrelationBalanceValidator:
    """Validate selected pairwise relationships between synthetic and source data."""

    def __init__(
        self,
        db_manager: DatabaseManager,
        config: ConfigManager,
        source_table: Optional[str] = None,
    ):
        self.db = db_manager
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.source_table = self._resolve_source_table(source_table)
        self.tolerance = float(config.get("validation.correlation_tolerance", 0.08))
        self.default_pairs = self._default_pairs()
        self.pairs = self._load_pairs_config(self.default_pairs)

    def validate_pairwise_correlations(
        self,
        sample_size: int = 100_000,
        synthetic_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """Run pairwise correlation balance validation."""
        source = self._load_source_sample(sample_size)
        synthetic = self._load_synthetic_sample(sample_size, synthetic_df)

        if source.empty or synthetic.empty:
            return {
                "success": False,
                "reason": "No source or synthetic sample available",
                "sample_sizes": {
                    "source": int(len(source)),
                    "synthetic": int(len(synthetic)),
                },
                "pair_results": [],
                "overall_score": 0.0,
            }

        source, synthetic = self._enrich_context_columns(source), self._enrich_context_columns(synthetic)

        pair_results: List[Dict[str, Any]] = []
        weighted_scores = []
        weighted_sum = 0.0
        diffs = []

        for pair_spec in self.pairs:
            result = self._evaluate_pair(pair_spec, source, synthetic)
            pair_results.append(result)
            status = result.get("status")
            if status != "ok":
                continue

            score = result.get("score")
            weight = float(result.get("weight", 1.0))
            if isinstance(score, (int, float)) and math.isfinite(score):
                weighted_scores.append(score * weight)
                weighted_sum += weight
                diffs.append(float(result.get("absolute_difference", 0.0)))

        overall = 0.0 if weighted_sum <= 0 else sum(weighted_scores) / weighted_sum

        return {
            "success": True,
            "sample_sizes": {
                "source": int(len(source)),
                "synthetic": int(len(synthetic)),
            },
            "pair_results": pair_results,
            "overall_score": float(overall),
            "mean_absolute_difference": float(np.mean(diffs)) if diffs else 0.0,
            "max_absolute_difference": float(np.max(diffs)) if diffs else 0.0,
            "tolerance": self.tolerance,
        }

    def _evaluate_pair(
        self,
        pair_spec: Dict[str, Any],
        source: pd.DataFrame,
        synthetic: pd.DataFrame,
    ) -> Dict[str, Any]:
        left = pair_spec["left"]
        right = pair_spec["right"]
        method = pair_spec.get("method", "auto").lower()
        name = pair_spec.get("name", f"{left}_{right}")
        weight = float(pair_spec.get("weight", 1.0))

        result: Dict[str, Any] = {
            "name": name,
            "left": left,
            "right": right,
            "method": method,
            "weight": weight,
            "status": "ok",
        }

        if left not in source.columns or right not in source.columns:
            return {
                **result,
                "status": "skipped",
                "reason": "pair columns missing in source",
            }
        if left not in synthetic.columns or right not in synthetic.columns:
            return {
                **result,
                "status": "skipped",
                "reason": "pair columns missing in synthetic",
            }

        source_left, source_right = source[left], source[right]
        synthetic_left, synthetic_right = synthetic[left], synthetic[right]

        source_left, source_right = self._coerce_for_analysis(source_left, source_right, method)
        synthetic_left, synthetic_right = self._coerce_for_analysis(synthetic_left, synthetic_right, method)

        source_strength = self._measure_pair_strength(source_left, source_right, method)
        synthetic_strength = self._measure_pair_strength(synthetic_left, synthetic_right, method)
        if source_strength is None or synthetic_strength is None:
            return {
                **result,
                "status": "skipped",
                "reason": "insufficient data for pair",
            }

        diff = float(abs(source_strength - synthetic_strength))
        score = self._scale_score(diff, self.tolerance)
        return {
            **result,
            "status": "ok",
            "source_strength": float(source_strength),
            "synthetic_strength": float(synthetic_strength),
            "absolute_difference": diff,
            "score": score,
        }

    def _measure_pair_strength(
        self,
        left: pd.Series,
        right: pd.Series,
        method: str,
    ) -> Optional[float]:
        method = method.lower()
        if method == "pearson":
            return self._pearson_strength(left, right)
        if method == "cramers_v":
            return self._cramers_v(left, right)
        if method == "eta":
            return self._correlation_ratio(left, right)
        if method == "auto":
            left_num = pd.api.types.is_numeric_dtype(left.dropna())
            right_num = pd.api.types.is_numeric_dtype(right.dropna())
            if left_num and right_num:
                return self._pearson_strength(left, right)
            if left_num != right_num:
                cat, num = (right, left) if left_num else (left, right)
                return self._correlation_ratio(cat.astype(str), num.astype(float))
            return self._cramers_v(left.astype(str), right.astype(str))

        if method == "ratio":
            return self._correlation_ratio(left, right)
        return None

    def _coerce_for_analysis(
        self,
        left: pd.Series,
        right: pd.Series,
        method: str,
    ) -> Tuple[pd.Series, pd.Series]:
        left = left.copy()
        right = right.copy()
        method = method.lower()

        if method in {"pearson"}:
            left = pd.to_numeric(left, errors="coerce")
            right = pd.to_numeric(right, errors="coerce")
            return left, right
        if method == "eta":
            if not pd.api.types.is_numeric_dtype(left):
                left = left.astype(str)
            else:
                left = left.astype(float)
            if not pd.api.types.is_numeric_dtype(right):
                right = right.astype(str)
            else:
                right = right.astype(float)
            return left, right

        # cramers_v and auto defaults to categorical handling
        return left.astype(str), right.astype(str)

    def _pearson_strength(self, left: pd.Series, right: pd.Series) -> Optional[float]:
        data = pd.DataFrame({"left": left, "right": right}).dropna()
        if len(data) < 100:
            return None
        corr = data["left"].astype(float).corr(data["right"].astype(float))
        if not np.isfinite(corr):
            return None
        return abs(float(corr))

    def _cramers_v(self, left: pd.Series, right: pd.Series) -> Optional[float]:
        data = pd.DataFrame({"left": left, "right": right}).dropna()
        if data.empty:
            return None
        table = pd.crosstab(data["left"], data["right"])
        if table.empty or table.shape[0] < 2 or table.shape[1] < 2:
            return 0.0
        chi2, _, _, _ = chi2_contingency(table, correction=False)
        n = table.to_numpy().sum()
        if n <= 0:
            return 0.0
        min_dim = min(table.shape[0] - 1, table.shape[1] - 1)
        if min_dim <= 0:
            return 0.0
        return float(math.sqrt((chi2 / n) / min_dim))

    def _correlation_ratio(self, categories: pd.Series, values: pd.Series) -> Optional[float]:
        data = pd.DataFrame({"category": categories, "value": values}).dropna()
        if data.empty:
            return None
        data["value"] = pd.to_numeric(data["value"], errors="coerce")
        data = data.dropna()
        if data.empty or data["value"].nunique() < 2:
            return None

        overall_mean = data["value"].mean()
        ss_total = ((data["value"] - overall_mean) ** 2).sum()
        if ss_total <= 0:
            return 0.0

        group_sums = data.groupby("category")["value"].agg(["mean", "count"])
        ss_between = ((group_sums["mean"] - overall_mean) ** 2 * group_sums["count"]).sum()
        eta2 = ss_between / ss_total
        eta2 = max(0.0, min(1.0, eta2))
        return float(math.sqrt(eta2))

    def _load_source_sample(self, sample_size: int) -> pd.DataFrame:
        if not self.source_table:
            return pd.DataFrame()
        needed = self._all_columns_needed()
        select_cols = ", ".join(sorted(needed))
        query = f"SELECT {select_cols} FROM {self.source_table} USING SAMPLE {sample_size}"
        try:
            data = self.db.fetch_dataframe(query)
        except Exception:
            return pd.DataFrame()
        return data.copy()

    def _load_synthetic_sample(
        self,
        sample_size: int,
        synthetic_df: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        needed = self._all_columns_needed()
        if synthetic_df is not None:
            available = [c for c in needed if c in synthetic_df.columns]
            if not available:
                return pd.DataFrame()
            if len(synthetic_df) > sample_size:
                synthetic_df = synthetic_df.sample(n=sample_size, random_state=42)
            return synthetic_df[available].copy()

        query = f"SELECT {', '.join(sorted(needed))} FROM nedis_synthetic.clinical_records USING SAMPLE {sample_size}"
        try:
            return self.db.fetch_dataframe(query).copy()
        except Exception:
            return pd.DataFrame()

    def _enrich_context_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        if {"vst_dt"}.issubset(result.columns) and "month" not in result.columns:
            dt = pd.to_datetime(result["vst_dt"].astype(str), format="%Y%m%d", errors="coerce")
            result["month"] = dt.dt.month.fillna(0).astype(int)
            result["dow"] = dt.dt.dayofweek.fillna(0).astype(int)
        if "vst_tm" in result.columns and "hour" not in result.columns:
            tm = result["vst_tm"].astype(str).str.zfill(4).str.slice(0, 4)
            result["hour"] = tm.str[:2].astype(int)
            result["minute"] = tm.str[2:4].astype(int)
        return result

    def _all_columns_needed(self) -> set:
        columns = set()
        for pair in self.pairs:
            columns.add(str(pair.get("left")))
            columns.add(str(pair.get("right")))
        for col in ["vst_dt", "vst_tm"]:
            columns.add(col)
        return columns

    @staticmethod
    def _scale_score(diff: float, tolerance: float) -> float:
        if not math.isfinite(diff) or not math.isfinite(tolerance) or tolerance <= 0:
            return 0.0
        return float(max(0.0, 1.0 - min(diff / tolerance, 1.0)))

    def _load_pairs_config(self, default_pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        configured = self.config.get("validation.correlation_pairs")
        if not configured:
            return default_pairs
        if not isinstance(configured, list):
            return default_pairs
        result = []
        for pair in configured:
            if not isinstance(pair, dict):
                continue
            left = pair.get("left")
            right = pair.get("right")
            if not left or not right:
                continue
            try:
                weight = float(pair.get("weight", 1.0))
            except (TypeError, ValueError):
                weight = 1.0
            item = {
                "name": str(pair.get("name", f"{left}_{right}")),
                "left": str(left),
                "right": str(right),
                "method": str(pair.get("method", "auto")),
                "weight": weight,
            }
            result.append(item)
        return result or default_pairs

    @staticmethod
    def _default_pairs() -> List[Dict[str, Any]]:
        return [
            {"name": "ktas_age", "left": "ktas_fstu", "right": "pat_age_gr", "method": "cramers_v", "weight": 1.0},
            {"name": "ktas_sex", "left": "ktas_fstu", "right": "pat_sex", "method": "cramers_v", "weight": 1.0},
            {"name": "age_hour", "left": "pat_age_gr", "right": "hour", "method": "auto", "weight": 1.0},
            {"name": "sex_hour", "left": "pat_sex", "right": "hour", "method": "auto", "weight": 0.8},
            {"name": "ktas_hour", "left": "ktas_fstu", "right": "hour", "method": "auto", "weight": 1.0},
            {"name": "month_hour", "left": "month", "right": "hour", "method": "pearson", "weight": 0.8},
            {"name": "month_dow", "left": "month", "right": "dow", "method": "pearson", "weight": 0.6},
        ]

    def _resolve_source_table(self, source_table: Optional[str] = None) -> Optional[str]:
        if source_table and isinstance(source_table, str):
            candidate = source_table.strip()
            if candidate:
                return candidate
        configured = self.config.get('original.source_table')
        if isinstance(configured, str) and configured.strip():
            return configured.strip()
        return None
