"""Collect results and build tables.  Results are CSV files (older runs may hold .parquet; both are read).

  python -m cpbench.results.aggregate cpbench_runs cpbench_results/results.csv
"""
import glob, os, sys
import pandas as pd


def read_result(stem):
    """One result table by path stem (stem.csv, or a legacy stem.parquet); None if absent."""
    if os.path.exists(stem + ".csv"):
        return pd.read_csv(stem + ".csv")
    if os.path.exists(stem + ".parquet"):
        return pd.read_parquet(stem + ".parquet")
    return None


def collect(root="cpbench_runs", protocol="audit_v2"):
    """Collect one scoring protocol; use protocol="legacy" for historical files. Never pool versions."""
    stems = {os.path.splitext(f)[0] for ext in ("csv", "parquet")
             for f in glob.glob(os.path.join(root, "**", f"*.{ext}"), recursive=True)}
    dfs = [read_result(s) for s in sorted(stems)]
    if not dfs:
        return pd.DataFrame()
    result = pd.concat(dfs, ignore_index=True)
    versions = result.get("scoring_protocol", pd.Series("legacy", index=result.index)).fillna("legacy")
    if protocol is None:
        if versions.nunique() > 1:
            raise ValueError("Mixed scoring protocols; collect each protocol separately")
        return result
    return result.loc[versions == protocol].reset_index(drop=True)


def table(df, metric="acc", detector="hmm", view="features", by=("features", "W", "gap"), columns="dataset"):
    """Mean and sd over seeds of one metric under one detector chosen in advance (never per dataset with labels).
    Returns (mean, sd); columns=None gives a single-column table."""
    d = df[(df.metric == metric) & (df.detector == detector) & (df.view == view)]
    if columns:
        g = d.groupby([*by, columns]).value
        return g.mean().unstack().round(3), g.std().unstack().round(3)
    g = d.groupby(list(by)).value
    return g.mean().round(3).to_frame(metric), g.std().round(3).to_frame(metric)


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "cpbench_runs"
    out = sys.argv[2] if len(sys.argv) > 2 else "cpbench_results/results.csv"
    df = collect(root); os.makedirs(os.path.dirname(out) or ".", exist_ok=True); df.to_csv(out, index=False)
    print(f"{len(df)} rows from {df.dataset.nunique() if len(df) else 0} datasets -> {out}")
