"""
13_excel.py
-----------
Writes the results table as a formatted Excel workbook.

The numbers come from 11_metrics_table.py's collect(), not from a second
reading of the summaries, so this file cannot drift from the printed table and
the LaTeX tabular. If a number here disagrees with the paper, the bug is in
collect() and all three outputs are wrong together, which is the failure mode
worth having.

Three sheets:

    Results   the table, grouped headers, best value in each column marked
    Charts    detection at matched false alarms, and against throughput
    Notes     what is measured, and why only some columns are ranked

Run:    python src/13_excel.py --pool 54
Output: Manuscripts/tables/results_pool<N>.xlsx
"""

from pathlib import Path
import argparse
import importlib.util
import sys

from openpyxl import Workbook
from openpyxl.chart import BarChart, Reference, ScatterChart, Series
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from lib_arms import DEFAULT_POOL  # noqa: E402

ROOT = SRC.parent
OUT = ROOT / "Manuscripts" / "tables"


def load_table_module():
    """Import 11_metrics_table.py, whose name is not a legal identifier."""
    spec = importlib.util.spec_from_file_location(
        "metrics_table", SRC / "11_metrics_table.py")
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(SRC))
    spec.loader.exec_module(mod)
    return mod


# (group, header, key, number format, higher is better)
# None in the last field means the column is descriptive, not ranked.
COLUMNS = [
    ("", "Model", "model", None, None),
    # The comparable columns: every arm read at the same false-alarm budget.
    # These are the only accuracy columns ranked.
    ("Detection at matched false alarms", "@5% FA", "det_fa05", "0.0", True),
    ("Detection at matched false alarms", "@10% FA", "det_fa10", "0.0", True),
    ("Detection at matched false alarms", "@15% FA", "det_fa15", "0.0", True),
    # Each arm at its own synthetic-chosen threshold. Not ranked: the
    # thresholds sit at different points of each arm's curve, so a leader here
    # is an arm whose threshold landed favourably, not a better model. Ranking
    # the false-alarm column alone rewarded exactly that.
    ("At each arm's synthetic threshold", "Screening detection %", "detect_rate", "0.0", None),
    ("At each arm's synthetic threshold", "False alarm %", "fa_rate", "0.0", None),
    ("At each arm's synthetic threshold", "Precision", "precision", "0.000", None),
    ("At each arm's synthetic threshold", "Recall", "recall", "0.000", None),
    ("At each arm's synthetic threshold", "F1", "f1", "0.000", None),
    ("Upper bound", "Oracle detection %", "oracle_detect", "0.0", None),
    ("Synthetic validation", "mAP@50", "map50", "0.000", None),
    ("Synthetic validation", "mAP@50:95", "map50_95", "0.000", None),
    ("Cost", "Latency ms", "latency_ms", "0.0", False),
    ("Cost", "FPS", "fps", "0.0", True),
    ("Cost", "Parameters M (fused)", "params_m", "0.00", False),
    ("Cost", "GFLOPs", "gflops", "0.0", False),
    ("Cost", "Weights MB", "size_mb", "0.0", False),
    ("Training", "Epochs run", "epochs_run", "0", None),
    ("Training", "Best epoch", "best_epoch", "0", None),
    ("Training", "Training min", "train_min", "0.0", None),
]

INK = "1F3B33"
HEAD_FILL = PatternFill("solid", fgColor="1F3B33")
GROUP_FILL = PatternFill("solid", fgColor="DCE8E2")
BEST_FILL = PatternFill("solid", fgColor="D6EAD9")
ENSEMBLE_FILL = PatternFill("solid", fgColor="F2E2D2")
THIN = Side(style="thin", color="C3CEC5")


def write_results(ws, rows):
    ws.freeze_panes = "B3"

    # Row 1 merges each group label across the columns it owns.
    col = 1
    while col <= len(COLUMNS):
        group = COLUMNS[col - 1][0]
        span = 1
        while col + span <= len(COLUMNS) and COLUMNS[col + span - 1][0] == group:
            span += 1
        if group:
            ws.merge_cells(start_row=1, start_column=col,
                           end_row=1, end_column=col + span - 1)
            c = ws.cell(row=1, column=col, value=group)
            c.font = Font(bold=True, size=9, color=INK)
            c.fill = GROUP_FILL
            c.alignment = Alignment(horizontal="center")
        col += span

    for i, (_, head, _, _, _) in enumerate(COLUMNS, start=1):
        c = ws.cell(row=2, column=i, value=head)
        c.font = Font(bold=True, color="FFFFFF", size=10)
        c.fill = HEAD_FILL
        c.alignment = Alignment(horizontal="center", vertical="center",
                                wrap_text=True)
        c.border = Border(bottom=THIN)
    ws.row_dimensions[2].height = 32

    for r, row in enumerate(rows, start=3):
        is_ens = row["model"] == "Ensemble"
        for i, (_, _, key, fmt, _) in enumerate(COLUMNS, start=1):
            v = row.get(key)
            c = ws.cell(row=r, column=i, value="-" if v is None else v)
            c.border = Border(bottom=THIN)
            if fmt and v is not None:
                # a lower bound keeps its number, so it still sorts and ranks,
                # but displays with a >= sign
                c.number_format = ('"\u2265"' + fmt if key in row.get("censored", ())
                                   else fmt)
            if i == 1:
                c.font = Font(bold=True, size=10)
            else:
                c.alignment = Alignment(horizontal="right")
            if is_ens:
                c.fill = ENSEMBLE_FILL

    # Mark the leader in every ranked column. Ties all get marked, which is
    # the honest rendering -- four points of difference is inside noise here.
    for i, (_, _, key, _, higher) in enumerate(COLUMNS, start=1):
        if higher is None:
            continue
        vals = [r.get(key) for r in rows if r.get(key) is not None]
        if not vals:
            continue
        target = max(vals) if higher else min(vals)
        for r, row in enumerate(rows, start=3):
            if row.get(key) == target:
                c = ws.cell(row=r, column=i)
                c.fill = BEST_FILL
                c.font = Font(bold=True, size=10)

    ws.column_dimensions["A"].width = 26
    for i in range(2, len(COLUMNS) + 1):
        ws.column_dimensions[get_column_letter(i)].width = 12
    ws.auto_filter.ref = f"A2:{get_column_letter(len(COLUMNS))}{len(rows) + 2}"


def write_charts(wb, ws_data, rows):
    ws = wb.create_sheet("Charts")
    n = len(rows)
    names = Reference(ws_data, min_col=1, min_row=3, max_row=n + 2)

    def col_of(key):
        for i, (_, _, k, _, _) in enumerate(COLUMNS, start=1):
            if k == key:
                return i
        raise KeyError(key)

    # Every arm at the same three false-alarm budgets: the comparison itself.
    bar = BarChart()
    bar.type, bar.style, bar.height, bar.width = "col", 10, 9, 22
    bar.title = "Detection at matched false-alarm budgets (comparable)"
    bar.y_axis.title = "detection %"
    for key in ("det_fa05", "det_fa10", "det_fa15"):
        c = col_of(key)
        bar.add_data(Reference(ws_data, min_col=c, min_row=2, max_row=n + 2),
                     titles_from_data=True)
    bar.set_categories(names)
    ws.add_chart(bar, "A1")

    # Accuracy against speed. One point per model; the ensemble sits far left.
    sc = ScatterChart()
    sc.style, sc.height, sc.width = 13, 9, 22
    sc.title = "Detection at 10% false alarms against throughput"
    sc.x_axis.title = "FPS"
    sc.y_axis.title = "detection % @10% FA"
    xs = Reference(ws_data, min_col=col_of("fps"), min_row=3, max_row=n + 2)
    ys = Reference(ws_data, min_col=col_of("det_fa10"), min_row=2, max_row=n + 2)
    s = Series(ys, xs, title_from_data=True)
    s.marker.symbol = "circle"
    s.graphicalProperties.line.noFill = True
    sc.series.append(s)
    ws.add_chart(sc, "A20")


NOTES = [
    ("How to read this workbook", True),
    ("", False),
    ("What is measured", True),
    ("Image-level screening. A photograph containing e-waste counts as "
     "detected when the model raises any detection in it; a photograph of "
     "organic waste counts as a false alarm when it does. Where the box lands "
     "is not measured and not claimed.", False),
    ("", False),
    ("Detection at matched false alarms", True),
    ("The columns to compare architectures on. Every arm is read at the same "
     "false-alarm budget -- the best detection rate its sweep reaches without "
     "exceeding 5, 10 or 15% false alarms -- so differences are attributable "
     "to the model. The budget is met using the test set's own false-alarm "
     "rate, so this is a comparison protocol, not an operating point anyone "
     "could have set in advance.", False),
    ("", False),
    ("Why the synthetic-threshold columns are not a comparison", True),
    ("Each arm's threshold is the argmax of its F1 curve on synthetic "
     "validation. Those curves are nearly flat -- 0.10 to 0.30 wide within "
     "0.02 of the peak, even on leak-free real data -- so where the argmax "
     "lands is close to arbitrary. An arm whose threshold landed low looks "
     "strong on detection and weak on false alarms, and the reverse. These "
     "columns are what a deployment without real calibration data would get, "
     "and are not ranked.", False),
    ("", False),
    ("Oracle detection %", True),
    ("Detection at the threshold that maximises F1 on the test set itself. An "
     "upper bound that assumes the answer is already known, not a result.",
     False),
    ("", False),
    ("Precision, recall and F1", True),
    ("These depend on the ratio of positives to negatives in the test "
     "split, an artefact of how it was drawn rather than a real rate of "
     "contamination. Detection and false-alarm rate do not.", False),
    ("", False),
    ("mAP columns", True),
    ("Measured on synthetic validation composites, not on real photographs. "
     "They describe box quality on the training distribution and are not "
     "ranked.", False),
    ("", False),
    ("Training", True),
    ("Every arm trains on one fixed schedule -- the same number of epochs, "
     "batch 16, seed 0, no early stopping -- and keeps the checkpoint that "
     "scored best on held-out synthetic validation. The learning rate decays "
     "linearly to the last epoch and mosaic augmentation switches off for the "
     "final 15, so a best epoch near the end is the schedule working as "
     "designed, not a sign the run was cut short.", False),
    ("", False),
    ("Cost", True),
    ("Latency is measured for every arm in one interleaved sitting and "
     "reported as the fastest round, since contention only adds time. The "
     "ranking is reproducible across sittings; the absolute level moves about "
     "5% with machine state. FLOPs do not predict latency on this hardware: "
     "the arm with the fewest GFLOPs is among the slowest, because "
     "depthwise-separable convolutions are bandwidth-bound. Parameter counts "
     "are after Conv-BN fusion, about 0.2% below the figures Ultralytics "
     "quotes.", False),
    ("", False),
    ("Ensemble", True),
    ("Weighted box fusion over all members, weights from synthetic "
     "validation F1. Those F1 values are nearly identical across arms, so the "
     "weights are close to uniform and the ensemble is effectively "
     "unweighted. Its cost row is the sum over members.", False),
    ("", False),
    ("Highlighting", True),
    ("A shaded cell leads its column. Only the matched-false-alarm and cost "
     "columns are ranked. Differences under about four points are inside the "
     "Wilson intervals of this test set; treat them as ties.", False),
]


def write_notes(wb):
    ws = wb.create_sheet("Notes")
    ws.column_dimensions["A"].width = 110
    for r, (text, is_head) in enumerate(NOTES, start=1):
        c = ws.cell(row=r, column=1, value=text)
        c.alignment = Alignment(wrap_text=True, vertical="top")
        if is_head:
            c.font = Font(bold=True, color=INK, size=11)
        else:
            ws.row_dimensions[r].height = 30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, default=DEFAULT_POOL)
    args = ap.parse_args()

    mt = load_table_module()
    # The same interleaved latency the printed table and LaTeX use, so the
    # workbook cannot show different cost figures from the paper.
    latency = mt.load_latency(args.pool)
    rows, missing = [], []
    for label, suffix in mt.MODELS:
        row = mt.collect(args.pool, label, suffix, latency)
        if row:
            rows.append(row)
        else:
            missing.append(label)

    if not rows:
        print(f"[!] no summaries for pool {args.pool}; run 06_evaluate.py first.")
        return
    if missing:
        print(f"[!] not evaluated, omitted: {', '.join(missing)}")

    wb = Workbook()
    ws = wb.active
    ws.title = "Results"
    write_results(ws, rows)
    write_charts(wb, ws, rows)
    write_notes(wb)

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"results_pool{args.pool}.xlsx"
    wb.save(path)
    print(f"written: {path.relative_to(ROOT)}  ({len(rows)} models)")


if __name__ == "__main__":
    main()
