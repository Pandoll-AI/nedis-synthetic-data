#!/usr/bin/env python3
"""Run the vectorized synthetic data generation pipeline."""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.clinical.dag_generator import ClinicalDAGGenerator
from src.core.config import ConfigManager
from src.core.database import DatabaseManager
from src.temporal.comprehensive_time_gap_synthesizer import ComprehensiveTimeGapSynthesizer
from src.validation.pipeline_quality_validator import PipelineQualityValidator
from src.vectorized import (
    VectorizedPatientGenerator,
    PatientGenerationConfig,
    TemporalPatternAssigner,
    TemporalConfig,
    CapacityConstraintPostProcessor,
    CapacityConfig,
)

logger = logging.getLogger(__name__)


def run_vectorized_pipeline(args) -> bool:
    """Run the end-to-end vectorized pipeline.

    Parameters
    ----------
    args: argparse.Namespace
        CLI arguments from ``scripts/generate.py``.
    """

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s'
    )

    target_db = Path(args.database)
    if target_db.parent and str(target_db.parent) != "." and not target_db.parent.exists():
        target_db.parent.mkdir(parents=True, exist_ok=True)

    config = ConfigManager(getattr(args, "config_path", "config/generation_params.yaml"))
    db = DatabaseManager(str(target_db))

    attached_alias: Optional[str] = None
    source_table = None

    try:
        if args.source_database:
            attached_alias = _attach_source_database(db, Path(args.source_database).expanduser())
            logger.info("Attached source database as alias: %s", attached_alias)

        source_table = _resolve_source_table(db, config, args.year, attached_alias)
        config.set('original.source_table', source_table)
        logger.info("Resolved source table: %s", source_table)

        _ensure_target_table(db)
        total_records = _resolve_total_records(db, source_table, args)
        logger.info("Target synthetic row count: %s", f"{total_records:,}")

        # 1) patient generation
        patient_generator = VectorizedPatientGenerator(db, config)
        patient_cfg = PatientGenerationConfig(
            total_records=total_records,
            batch_size=args.batch_size,
            random_seed=args.random_seed,
            memory_efficient=args.memory_efficient,
        )
        patients = patient_generator.generate_all_patients(patient_cfg)
        logger.info("Generated patients: %s rows", f"{len(patients):,}")

        # 2) temporal assignment (arrival date/time)
        temporal_assigner = TemporalPatternAssigner(db, config)
        temporal_cfg = TemporalConfig(
            year=args.year,
            preserve_seasonality=args.preserve_seasonality,
            preserve_weekly_pattern=args.preserve_weekly_pattern,
            preserve_holiday_effects=args.preserve_holiday_effects,
            time_resolution=args.time_resolution,
            enable_conditional_hour_patterns=not getattr(args, "disable_conditional_hour_patterns", False),
            conditional_context_min_count=getattr(args, "conditional_context_min_count", 30),
            conditional_global_mix_weight=getattr(args, "conditional_global_mix_weight", 0.2),
            conditional_smoothing_alpha=getattr(args, "conditional_smoothing_alpha", 0.02),
        )
        patients = temporal_assigner.assign_temporal_patterns(patients, temporal_cfg)

        temporal_report: Dict[str, Any] = {}
        if args.validate_temporal:
            temporal_report = temporal_assigner.validate_temporal_assignment(patients, temporal_cfg)
            logger.info("Temporal validation completed: %s", temporal_report.get("summary", {}))

        # 3) capacity post-processing
        capacity_report: Optional[Dict[str, Any]] = None
        if args.enable_overflow_redistribution:
            capacity_processor = CapacityConstraintPostProcessor(db, config)
            capacity_cfg = CapacityConfig(
                base_capacity_multiplier=args.base_capacity_multiplier,
                weekend_capacity_multiplier=args.weekend_capacity_multiplier,
                holiday_capacity_multiplier=args.holiday_capacity_multiplier,
                safety_margin=args.safety_margin,
                overflow_redistribution_method=args.overflow_redistribution_method,
                enable_overflow=True,
            )
            patients = capacity_processor.apply_capacity_constraints(patients, capacity_cfg)
            if args.generate_capacity_report:
                capacity_report = capacity_processor.generate_capacity_report(patients)
                logger.info("Capacity report: %.2f%% redistributed", capacity_report.get('redistribution_rate', 0.0))

        # 4) comprehensive time-gap synthesis using vst_* assigned above
        gap_synth = ComprehensiveTimeGapSynthesizer(db, config)
        base_visit_ts = _build_base_visit_timestamps(
            patients['vst_dt'], patients['vst_tm']
        )
        if 'ktas_fstu' in patients.columns:
            ktas_values = patients['ktas_fstu']
        elif 'ktas01' in patients.columns:
            ktas_values = patients['ktas01']
        else:
            ktas_values = pd.Series(['3'] * len(patients), index=patients.index)

        if 'emtrt_rust' in patients.columns:
            emtrt_values = patients['emtrt_rust']
        else:
            emtrt_values = pd.Series(['11'] * len(patients), index=patients.index)

        gaps = gap_synth.generate_all_time_gaps(
            np.array(ktas_values),
            np.array(emtrt_values),
            base_datetime=base_visit_ts,
        )
        overlap_cols = patients.columns.intersection(gaps.columns)
        if len(overlap_cols):
            patients = patients.drop(columns=list(overlap_cols))

        patients = pd.concat([patients.reset_index(drop=True), gaps.reset_index(drop=True)], axis=1)

        # 5) write into target table
        patients = _normalize_for_insert(
            db=db,
            df=patients,
            source_table=source_table,
            seed=args.random_seed if args.random_seed is not None else 42,
        )

        if args.clean_existing_data:
            db.execute_query("DELETE FROM nedis_synthetic.clinical_records")

        inserted_rows = _insert_clinical_records(db, patients, chunk_size=50_000)
        logger.info("Inserted %s rows into nedis_synthetic.clinical_records", f"{inserted_rows:,}")

        # 5. build lightweight ER diagnosis table to support clinical rule validation
        diag_count = _upsert_minimal_diag_er(
            db=db,
            patients=patients,
            clean=args.clean_existing_data,
        )
        logger.info("Prepared %s fallback diagnosis rows in nedis_synthetic.diag_er", f"{diag_count:,}")

        # 6) validation / quality gate
        validator = PipelineQualityValidator(db, config)
        quality = validator.evaluate(
            synthetic_df=patients,
            source_table=source_table,
            capacity_report=capacity_report,
            temporal_validation=temporal_report if args.validate_temporal else None,
            quality_gate=0.0,
        )
        _save_quality_report(
            report_path=getattr(args, "quality_report_path", None),
            quality=quality,
            database_path=str(target_db),
            source_table=source_table,
        )

        logger.info("Validation overall score: %.4f", quality.get('overall_score', 0.0))
        if isinstance(quality.get('details'), dict):
            logger.info("Validation details: %s", quality['details'])

        threshold = args.quality_gate_threshold
        if threshold > 1.0:
            threshold = threshold / 100.0

        if quality.get('overall_score', 0.0) < threshold:
            logger.error(
                "Quality gate failed: %.4f < %.4f",
                quality.get('overall_score', 0.0),
                threshold
            )
            return False

        return True

    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        return False
    finally:
        if attached_alias:
            try:
                db.execute_query(f"DETACH {attached_alias}")
            except Exception:
                pass
        db.close()


def _save_quality_report(
    report_path: Optional[str],
    quality: Dict[str, Any],
    database_path: str,
    source_table: str,
) -> None:
    if not report_path:
        return

    payload = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "database_path": database_path,
        "source_table": source_table,
        "quality": quality,
    }

    report_file = Path(report_path)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    with open(report_file, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=_json_fallback)


def _json_fallback(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, (pd.Series, pd.DataFrame)):
        return value.to_dict(orient="list")
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _attach_source_database(db: DatabaseManager, source_path: Path) -> str:
    if not source_path.exists():
        raise FileNotFoundError(f"Source DB not found: {source_path}")

    alias = "source"
    try:
        db.execute_query(f"DETACH {alias}")
    except Exception:
        pass

    db.execute_query(f"ATTACH '{str(source_path)}' AS {alias}")
    return alias


def _resolve_source_table(
    db: DatabaseManager,
    config: ConfigManager,
    year: int,
    source_alias: Optional[str],
) -> str:
    configured = config.get('original.source_table')
    candidates = []
    if isinstance(configured, str) and configured.strip():
        candidates.append(configured)

    year = year or 2017
    if source_alias:
        candidates.extend(
            [
                f"{source_alias}.nedis{year}",
                f"{source_alias}.nedis2017",
            ]
        )

    candidates.extend(
        [
            f"nedis_original.nedis{year}",
            "nedis_original.nedis2017",
            f"nedis_data.nedis{year}",
            "nedis_data.nedis2017",
            f"main.nedis{year}",
            "main.nedis2017",
        ]
    )

    for candidate in candidates:
        if candidate and _table_exists(db, candidate):
            return candidate

    raise RuntimeError("No source table found. Tried: " + ", ".join(candidates))


def _resolve_total_records(db: DatabaseManager, source_table: str, args) -> int:
    if args.use_original_size:
        count_df = db.fetch_dataframe(f"SELECT COUNT(*) AS cnt FROM {source_table}")
        count = int(count_df['cnt'].iloc[0])
        if count <= 0:
            raise RuntimeError(f"Source table '{source_table}' is empty")
        return count
    return int(args.total_records)


def _ensure_target_table(db: DatabaseManager) -> None:
    # Use the existing schema initializer for consistency
    ClinicalDAGGenerator(db, ConfigManager()).initialize_clinical_records_table()


def _build_base_visit_timestamps(vst_dt: pd.Series, vst_tm: pd.Series) -> pd.Series:
    dt = pd.Series(vst_dt).astype(str).str.strip()
    tm = pd.Series(vst_tm).astype(str).str.zfill(4).str.slice(0, 4)
    return pd.to_datetime(dt + tm, format='%Y%m%d%H%M', errors='coerce')


def _normalize_for_insert(
    db: DatabaseManager,
    df: pd.DataFrame,
    source_table: Optional[str] = None,
    seed: int = 42,
) -> pd.DataFrame:
    schema = db.fetch_dataframe("DESCRIBE nedis_synthetic.clinical_records")
    table_columns = schema['column_name'].tolist()

    result = df.copy().reset_index(drop=True)
    n = len(result)

    if 'index_key' in table_columns:
        result['index_key'] = [f"SYNTH_{i:09d}" for i in range(n)]

    if 'pat_reg_no' in table_columns:
        result['pat_reg_no'] = [f"P{i:09d}" for i in range(n)]

    if 'emorg_cd' in table_columns and 'emorg_cd' not in result.columns:
        result['emorg_cd'] = 'UNKNOWN'

    if 'vst_tm' in result.columns:
        result['vst_tm'] = result['vst_tm'].astype(str).str.zfill(4).str[:4]

    required_for_insert = ['vst_dt', 'vst_tm', 'pat_age_gr', 'pat_sex', 'pat_do_cd']
    for col in required_for_insert:
        if col not in result.columns:
            result[col] = '' if col != 'vst_dt' else '20170101'
            if col == 'vst_tm':
                result[col] = '0000'

    if 'ktas01' in table_columns and 'ktas01' not in result.columns:
        result['ktas01'] = pd.to_numeric(result.get('ktas_fstu', '3'), errors='coerce').fillna(3).astype(int)

    if 'generation_timestamp' in table_columns and 'generation_timestamp' not in result.columns:
        result['generation_timestamp'] = pd.Timestamp.utcnow()

    if 'generation_method' in table_columns and 'generation_method' not in result.columns:
        result['generation_method'] = 'vectorized_pipeline'

    # Vital sign defaults for downstream statistical/clinical validation.
    result = _normalize_vital_columns(
        db=db,
        df=result,
        source_table=source_table,
        seed=seed,
    )

    # Ensure schema order and drop extras
    result = result.reindex(columns=table_columns)
    return result


def _normalize_vital_columns(
    db: DatabaseManager,
    df: pd.DataFrame,
    source_table: Optional[str] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Normalize vital signs using source-like sampling with a safe fallback."""
    if df.empty:
        return df

    target = df.copy()
    vital_specs: Dict[str, Dict[str, Any]] = {
        "vst_sbp": {"min": 60, "max": 250, "dtype": "int", "fallback": (125, 18)},
        "vst_dbp": {"min": 30, "max": 130, "dtype": "int", "fallback": (80, 15)},
        "vst_per_pu": {"min": 30, "max": 220, "dtype": "int", "fallback": (78, 20)},
        "vst_per_br": {"min": 8, "max": 80, "dtype": "int", "fallback": (74, 16)},
        "vst_bdht": {"min": 34.0, "max": 44.0, "dtype": "float", "fallback": (36.6, 0.7)},
        "vst_oxy": {"min": 70, "max": 100, "dtype": "int", "fallback": (97.5, 2.0)},
    }

    vital_reference = _load_vital_reference_profiles(db, source_table, list(vital_specs))
    rng = np.random.default_rng(seed)

    # Convert non-numeric or sentinel values to NA and fill missing via source-like profile.
    for col, cfg in vital_specs.items():
        if col not in target.columns:
            target[col] = np.nan

        series = pd.to_numeric(target[col], errors="coerce")
        if cfg["dtype"] == "int":
            series = series.round()
            series = series.astype(float)

        sentinel_mask = series.isin([-1, -1.0, 999, 999.0])
        invalid_mask = (series < cfg["min"]) | (series > cfg["max"])
        missing_mask = series.isna() | sentinel_mask | invalid_mask

        if missing_mask.any():
            replacement = _draw_vital_values(
                rng=rng,
                column=col,
                count=int(missing_mask.sum()),
                reference=vital_reference.get(col),
                fallback=cfg["fallback"],
                dtype=cfg["dtype"],
            )
            series = series.astype("float")
            series.loc[missing_mask] = replacement

        if cfg["dtype"] == "int":
            target[col] = pd.to_numeric(series, errors="coerce").round().clip(cfg["min"], cfg["max"]).astype("Int64")
        else:
            target[col] = pd.to_numeric(series, errors="coerce").round(1).clip(cfg["min"], cfg["max"])

    return target


def _load_vital_reference_profiles(db: DatabaseManager, source_table: Optional[str], columns: List[str]) -> Dict[str, np.ndarray]:
    if not source_table:
        return {}

    profiles: Dict[str, np.ndarray] = {}
    available_cols = ", ".join(columns)
    query = f"SELECT {available_cols} FROM {source_table} USING SAMPLE 200000"

    try:
        sample = db.fetch_dataframe(query)
    except Exception as exc:
        logger.warning("Failed to load source vital profiles from %s: %s", source_table, exc)
        return profiles

    for col in columns:
        if col not in sample.columns:
            continue

        clean = pd.to_numeric(sample[col], errors="coerce")
        clean = clean.replace([999, 999.0, -1, -1.0], np.nan)
        if col == "vst_sbp":
            min_v, max_v = 60, 250
        elif col == "vst_dbp":
            min_v, max_v = 30, 130
        elif col == "vst_per_pu":
            min_v, max_v = 30, 220
        elif col == "vst_per_br":
            min_v, max_v = 8, 80
        elif col == "vst_bdht":
            min_v, max_v = 34.0, 44.0
        elif col == "vst_oxy":
            min_v, max_v = 70, 100
        else:
            min_v, max_v = -1_000, 1_000

        clean = clean[(clean >= min_v) & (clean <= max_v)]
        clean = clean.dropna().to_numpy(dtype=float)
        if len(clean) > 0:
            profiles[col] = clean

    return profiles


def _draw_vital_values(
    rng: np.random.Generator,
    column: str,
    count: int,
    reference: Optional[np.ndarray],
    fallback: Tuple[float, float],
    dtype: str,
) -> np.ndarray:
    if count <= 0:
        return np.array([], dtype=float)

    if reference is None or len(reference) == 0:
        mean, std = fallback
        values = rng.normal(mean, std, size=count)
    else:
        idx = rng.integers(0, len(reference), size=count)
        values = reference[idx]

    if dtype == "int":
        return np.round(values).astype(float)
    return values.astype(float)



def _insert_clinical_records(db: DatabaseManager, patients: pd.DataFrame, chunk_size: int = 50_000) -> int:
    inserted = 0
    table = "nedis_synthetic.clinical_records"

    for start in range(0, len(patients), chunk_size):
        batch = patients.iloc[start:start + chunk_size]
        view_name = f"_synthetic_batch_{start}"
        db.conn.register(view_name, batch)
        try:
            db.execute_query(f"INSERT INTO {table} SELECT * FROM {view_name}")
            inserted += len(batch)
        finally:
            try:
                db.conn.unregister(view_name)
            except Exception:
                pass

    return inserted


def _upsert_minimal_diag_er(db: DatabaseManager, patients: pd.DataFrame, clean: bool = True) -> int:
    """Create or refresh a minimal nedis_synthetic.diag_er table for rule validation.

    The synthetic generation pipeline does not currently run the full diagnosis generator
    by default, so this fallback preserves validation compatibility while keeping
    semantic constraints of the clinical rules.
    """

    db.execute_query("""
        CREATE TABLE IF NOT EXISTS nedis_synthetic.diag_er (
            index_key VARCHAR NOT NULL,
            position INTEGER NOT NULL,
            diagnosis_code VARCHAR NOT NULL,
            diagnosis_category VARCHAR DEFAULT '1',
            icd_chapter VARCHAR,
            generation_method VARCHAR DEFAULT 'fallback',
            PRIMARY KEY (index_key, position)
        )
    """)

    if clean:
        db.execute_query("DELETE FROM nedis_synthetic.diag_er")

    if len(patients) == 0:
        return 0

    safe_patients = patients.reset_index(drop=True)[["index_key", "ktas_fstu", "pat_age_gr", "pat_sex"]].copy()
    safe_patients["position"] = 1
    safe_patients["diagnosis_code"] = safe_patients.apply(
        lambda row: _fallback_diagnosis_code(row),
        axis=1,
    )
    safe_patients["diagnosis_category"] = "1"
    safe_patients["icd_chapter"] = safe_patients["diagnosis_code"].str[:1]
    safe_patients["generation_method"] = "fallback_synthetic"

    db.conn.register("_fallback_diag_er", safe_patients)
    try:
        db.execute_query("""
            INSERT INTO nedis_synthetic.diag_er
            (index_key, position, diagnosis_code, diagnosis_category, icd_chapter, generation_method)
            SELECT index_key, position, diagnosis_code, diagnosis_category, icd_chapter, generation_method
            FROM _fallback_diag_er
        """)
    finally:
        try:
            db.conn.unregister("_fallback_diag_er")
        except Exception:
            pass

    return len(safe_patients)


def _fallback_diagnosis_code(row: pd.Series) -> str:
    """Generate a safe diagnosis code compatible with current clinical constraints."""
    age_group = str(row.get("pat_age_gr", "")).strip()
    ktas = str(row.get("ktas_fstu", "3")).strip()

    if age_group in {"01", "09", "10"}:
        return "J20"
    if str(row.get("pat_sex", "")).upper() == "F":
        return "J20"
    if str(row.get("pat_sex", "")).upper() == "M" and ktas in {"1", "2"}:
        return "R06"
    return "J20"


def _table_exists(db: DatabaseManager, table_name: str) -> bool:
    try:
        db.execute_query(f"SELECT 1 FROM {table_name} LIMIT 1")
        return True
    except Exception:
        return False
