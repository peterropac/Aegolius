#!/usr/bin/env python3
"""run_tests.py — sanity-check SPOMSO by running every public example.

Usage:
    python run_tests.py           # run and check against stored snapshots
    python run_tests.py --update  # run and update snapshots (after intentional changes)
    python tests/run_tests.py --slow      # also run examples on the SLOW list
"""
import sys, os, runpy, json, argparse
from pathlib import Path
import numpy as np

# Headless setup
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.show = lambda *a, **k: plt.close("all")
try:
    import plotly.graph_objects as go
    go.Figure.show = lambda *a, **k: None
except ImportError:
    pass


SPOMSO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = SPOMSO_ROOT / "examples"
SNAPSHOTS = Path(__file__).resolve().parent / "test_snapshots"
COVERAGE = Path(__file__).resolve().parent / "coverage"

SLOW = ["autodiff/multi_position_optimization.py", "autodiff/position_optimization.py",
        "autodiff/vector_field_optimization.py"]

print("Examples directory: ", EXAMPLES)
print("Snapshots directory: ", SNAPSHOTS)
print("Coverage tests directory: ", COVERAGE)

# Optionally — which variable name in each example holds "the answer"
OUTPUT_NAMES = ["sdf_values", "field", "result", "components"]

def summarize(arr):
    arr = np.asarray(arr)
    return {
        "shape": list(arr.shape),
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
        "mean": float(np.nanmean(arr)),
    }

def find_examples(include_slow=False):
    if include_slow:
        return sorted(p for p in EXAMPLES.rglob("*.py") if p.with_suffix(".ipynb").exists())
    else:
        return sorted(p for p in EXAMPLES.rglob("*.py") if p.with_suffix(".ipynb").exists() and p.relative_to(EXAMPLES).as_posix() not in SLOW)


def find_jax_coverage_examples():
    out = sorted(p for p in COVERAGE.rglob("*.py") if p.name.startswith("jax_test_all_"))
    return out

def find_numpy_coverage_examples():
    out = sorted(p for p in COVERAGE.rglob("*.py") if p.name.startswith("test_all_"))
    return out

def run_one(path, update=False):
    rel = path.relative_to(EXAMPLES if path.is_relative_to(EXAMPLES) else COVERAGE)
    original_cwd = os.getcwd()
    os.chdir(path.parent)

    added = str(path.parent) not in sys.path
    if added:
        sys.path.insert(0, str(path.parent))

    try:
        ns = runpy.run_path(path.name, run_name="__main__")
    except Exception as e:
        return ("CRASH", f"{type(e).__name__}: {e}")
    finally:
        os.chdir(original_cwd)
        if added:
            sys.path.remove(str(path.parent))
    
    # Find the output variable
    output = next((ns[n] for n in OUTPUT_NAMES if n in ns), None)
    if output is None:
        return ("OK-NO-SNAPSHOT", "ran cleanly, no recognised output var")

    snap_path = SNAPSHOTS / rel.with_suffix('.json')
    summary = summarize(output)

    if update:
        snap_path.parent.mkdir(parents=True, exist_ok=True)
        snap_path.write_text(json.dumps(summary, indent=2))
        return ("UPDATED", "")

    if not snap_path.exists():
        return ("NO-SNAPSHOT", "no stored snapshot; run with --update")

    expected = json.loads(snap_path.read_text())
    for k in ("shape", "min", "max", "mean"):
        if k == "shape":
            if expected[k] != summary[k]:
                return ("FAIL", f"shape: {expected[k]} → {summary[k]}")
        elif not np.isclose(expected[k], summary[k], rtol=1e-5, atol=1e-5):
            return ("FAIL", f"{k}: {expected[k]:.6g} → {summary[k]:.6g}")
    return ("OK", "")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--update", action="store_true",
                    help="update stored snapshots instead of checking")
    ap.add_argument("--slow", action="store_true",
                    help="also run slow examples (default: skip)")
    args = ap.parse_args()

    if not args.slow and SLOW:
        print(f"(skipping {len(SLOW)} slow example(s); use --slow to include)\n")

    examples = find_examples(include_slow=args.slow)
    jax_coverage_examples = find_jax_coverage_examples()
    numpy_coverage_examples = find_numpy_coverage_examples()
    examples.extend(jax_coverage_examples)
    examples.extend(numpy_coverage_examples)

    results = []

    markers = {"OK": "✓", "OK-NO-SNAPSHOT": "○", "UPDATED": "↻",
                  "FAIL": "✗", "CRASH": "✗", "NO-SNAPSHOT": "?"}

    for p in examples:
        status, msg = run_one(p, update=args.update)
        results.append((p, status, msg))
        marker = markers[status]
        print(f"  {marker} {p.relative_to(p.parents[1])}  {msg}")

    n_fail = sum(1 for _, s, _ in results if s in ("FAIL", "CRASH"))
    n_pass = len(results) - n_fail
    print(f"\n{n_pass} passed, {n_fail} failed, {len(results)} total")
    sys.exit(1 if n_fail else 0)

if __name__ == "__main__":
    main()