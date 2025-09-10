#!/usr/bin/env python3
"""
Flask dashboard to compare original vs synthetic data side-by-side.

Features:
- Per-column comparison for categorical vars: value counts, proportions, chi-square p-values
- Temporal comparisons: hourly distribution, inter-arrival time distribution (minutes)
- Renders a 2-column HTML layout (Original | Synthetic)
- Supports export to a single static HTML file in outputs/
"""

import io
import base64
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from flask import Flask, render_template, request, send_file
import duckdb
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import argparse
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


DEFAULT_DB = 'nedis_sample.duckdb'
ORIG_TABLE = 'nedis_original.nedis2017'
SYN_TABLE = 'nedis_synthetic.clinical_records'


app = Flask(__name__, template_folder=str(Path(__file__).parent.parent / 'templates'))


def get_conn(db_path: str):
    return duckdb.connect(db_path)


def df_value_counts_norm(df: pd.DataFrame, col: str, top: int = 15) -> pd.DataFrame:
    if col not in df.columns:
        return pd.DataFrame(columns=[col, 'count', 'proportion'])
    vc = df[col].value_counts(dropna=False)
    if len(vc) == 0:
        return pd.DataFrame(columns=[col, 'count', 'proportion'])
    vc = vc.head(top)
    total = vc.sum()
    return pd.DataFrame({col: vc.index.astype(str), 'count': vc.values, 'proportion': (vc.values / total)})


def fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png', dpi=120)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('ascii')


def parse_timestamp(df: pd.DataFrame, date_col: str = 'vst_dt', time_col: str = 'vst_tm') -> pd.Series:
    if date_col not in df.columns or time_col not in df.columns:
        return pd.Series([], dtype='datetime64[ns]')
    dt = pd.to_datetime(df[date_col].astype(str), format='%Y%m%d', errors='coerce')
    tm = df[time_col].astype(str).str.pad(4, fillchar='0')
    hh = pd.to_numeric(tm.str[:2], errors='coerce')
    mm = pd.to_numeric(tm.str[2:], errors='coerce')
    dt = dt + pd.to_timedelta(hh.fillna(0), unit='h') + pd.to_timedelta(mm.fillna(0), unit='m')
    return dt


def inter_arrival_minutes(ts: pd.Series) -> pd.Series:
    ts = ts.dropna().sort_values()
    if len(ts) < 2:
        return pd.Series([], dtype=float)
    diffs = ts.diff().dropna().dt.total_seconds() / 60.0
    return diffs


def hourly_distribution(df: pd.DataFrame, time_col: str = 'vst_tm') -> pd.Series:
    if time_col not in df.columns:
        return pd.Series([], dtype=float)
    tm = df[time_col].astype(str).str.pad(4, fillchar='0')
    hour = pd.to_numeric(tm.str[:2], errors='coerce').dropna().astype(int)
    hist = hour.value_counts(normalize=True).sort_index()
    # ensure 0-23
    for h in range(24):
        if h not in hist.index:
            hist.loc[h] = 0.0
    return hist.sort_index()


def plot_bar_compare(categories: List[str], left: List[float], right: List[float], title: str, left_label: str, right_label: str):
    x = np.arange(len(categories))
    width = 0.4
    fig, ax = plt.subplots(figsize=(max(6, len(categories) * 0.4), 3.0))
    ax.bar(x - width / 2, left, width, label=left_label, alpha=0.8)
    ax.bar(x + width / 2, right, width, label=right_label, alpha=0.8)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Proportion')
    ax.legend()
    return fig


def plot_hist_compare(data_left: np.ndarray, data_right: np.ndarray, bins: int, title: str, left_label: str, right_label: str):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(data_left, bins=bins, density=True, alpha=0.6, label=left_label)
    ax.hist(data_right, bins=bins, density=True, alpha=0.6, label=right_label)
    ax.set_title(title)
    ax.set_ylabel('Density')
    ax.legend()
    return fig


def chi_square_pvalue(counts_a: pd.Series, counts_b: pd.Series) -> Optional[float]:
    # Align categories
    idx = sorted(set(counts_a.index) | set(counts_b.index))
    a = np.array([counts_a.get(i, 0) for i in idx])
    b = np.array([counts_b.get(i, 0) for i in idx])
    if a.sum() == 0 or b.sum() == 0:
        return None
    try:
        stat, p = stats.chisquare(a, f_exp=(b / b.sum()) * a.sum())
        return float(p)
    except Exception:
        return None


@app.route('/')
def index():
    db_path = request.args.get('db', DEFAULT_DB)
    conn = get_conn(db_path)

    # Load samples (limit for render speed)
    orig = conn.execute(f"SELECT * FROM {ORIG_TABLE} USING SAMPLE 200000").fetch_df()
    syn = conn.execute(f"SELECT * FROM {SYN_TABLE}").fetch_df()

    # Categorical columns to compare (if present)
    cat_cols = [
        'pat_age_gr', 'pat_sex', 'pat_do_cd', 'ktas_fstu', 'emtrt_rust',
        'vst_meth', 'msypt', 'main_trt_p', 'emorg_cd'
    ]

    comparisons = []
    for col in cat_cols:
        left_df = df_value_counts_norm(orig, col)
        right_df = df_value_counts_norm(syn, col)
        if left_df.empty and right_df.empty:
            continue
        # Bar chart data unify categories
        cats = sorted(set(left_df[col].astype(str)) | set(right_df[col].astype(str)))[:20]
        left_map = dict(zip(left_df[col].astype(str), left_df['proportion']))
        right_map = dict(zip(right_df[col].astype(str), right_df['proportion']))
        left_vals = [float(left_map.get(c, 0.0)) for c in cats]
        right_vals = [float(right_map.get(c, 0.0)) for c in cats]
        fig = plot_bar_compare(cats, left_vals, right_vals, f"{col} distribution", "Original", "Synthetic")
        img_b64 = fig_to_base64(fig)

        # chi-square p-value on counts
        pval = chi_square_pvalue(
            orig[col].astype(str).value_counts(), syn[col].astype(str).value_counts()
        ) if (col in orig.columns and col in syn.columns) else None

        comparisons.append({
            'column': col,
            'left_table': left_df.to_dict(orient='records'),
            'right_table': right_df.to_dict(orient='records'),
            'chart_png': img_b64,
            'chi_square_p': pval
        })

    # Temporal comparisons
    orig_ts = parse_timestamp(orig)
    syn_ts = parse_timestamp(syn)
    orig_ia = inter_arrival_minutes(orig_ts)
    syn_ia = inter_arrival_minutes(syn_ts)
    # cap extreme to reasonable for viz
    cap = 240
    fig_ia = plot_hist_compare(
        orig_ia.clip(upper=cap).values,
        syn_ia.clip(upper=cap).values,
        bins=30,
        title='Inter-arrival time (minutes, clipped at 240)',
        left_label='Original',
        right_label='Synthetic'
    )
    img_ia = fig_to_base64(fig_ia)

    # hourly distribution
    orig_hour = hourly_distribution(orig)
    syn_hour = hourly_distribution(syn)
    fig_hr = plot_bar_compare(
        [str(h) for h in range(24)],
        [float(orig_hour.get(h, 0.0)) for h in range(24)],
        [float(syn_hour.get(h, 0.0)) for h in range(24)],
        'Hourly arrival distribution', 'Original', 'Synthetic'
    )
    img_hr = fig_to_base64(fig_hr)

    return render_template(
        'comparison.html',
        db_path=db_path,
        comparisons=comparisons,
        interarrival_png=img_ia,
        hourly_png=img_hr,
        orig_count=len(orig),
        syn_count=len(syn)
    )


@app.route('/export')
def export_static():
    # Render the same page and save as outputs HTML
    html = index()
    out_dir = Path('outputs')
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / 'synthetic_vs_original_comparison.html'
    out_path.write_text(html, encoding='utf-8')
    return f"Saved: {out_path}"


def main():
    parser = argparse.ArgumentParser(description='Original vs Synthetic Comparison Dashboard (Flask)')
    parser.add_argument('--database', default=DEFAULT_DB, help='DuckDB path')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=5001)
    parser.add_argument('--export-only', action='store_true', help='Render and export a static HTML without running server')
    args = parser.parse_args()

    app.config['DB_PATH'] = args.database

    # Inject default DB param when no query parameter
    @app.before_request
    def attach_db():
        if not request.args.get('db'):
            request.args = request.args.copy()
            request.args = request.args.to_dict()
            # no-op: we keep default in index() via DEFAULT_DB
            return None

    if args.export_only:
        with app.test_request_context(f"/?db={args.database}"):
            html = index()
        out_dir = Path('outputs')
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / 'synthetic_vs_original_comparison.html'
        out_path.write_text(html, encoding='utf-8')
        print(f"Saved: {out_path}")
        return

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == '__main__':
    main()

