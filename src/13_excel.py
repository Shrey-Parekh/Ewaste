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
    Charts    detection against false alarms, F1, and accuracy against speed
    Notes     what the operating point is and why the oracle column exists

Run:    python src/13_excel.py --pool 60
Output: Manuscripts/tables/results_pool60.xlsx
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
    ("At each arm's own threshold", "Detection %", "detect_rate", "0.0", None),
    ("At each arm's own threshold", "False alarm %", "fa_rate", "0.0", None),
    ("At each arm's own threshold", "Precision", "precision", "0.000", None),
    ("At each arm's own threshold", "Recall", "recall", "0.000", None),
    ("At each arm's own threshold", "F1", "f1", "0.000", True),
    # The comparable columns. Every arm read at the same false-alarm budget.
    ("Detection at matched false alarms", "@5% FA", "det_fa05", "0.0", True),
    ("Detection at matched false alarms", "@10% FA", "det_fa10", "0.0", True),
    ("Detection at matched false alarms", "@15% FA", "det_fa15", "0.0", True),
    ("Synthetic validation", "mAP@50", "map50", "0.000", True),
    ("Synthetic validation", "mAP@50:95", "map50_95", "0.000", True),
    ("Localisation", "Hit % when fired", "hit_rate", "0.0", True),
    ("Localisation", "mIoU", "miou", "0.000", None),
    ("Localisation", "Dice", "dice", "0.000", None),
    ("Localisation", "n matched", "n_matched", "0", None),
    ("Upper bound", "Oracle detection %", "oracle_detect", "0.0", None),
    ("Cost", "Latency ms", "latency_ms", "0.0", False),
    ("Cost", "FPS", "fps", "0.0", True),
    ("Cost", "Parameters M", "params_m", "0.00", False),
    ("Cost", "GFLOPs", "gflops", "0.0", False),
    ("Cost", "Weights MB", "size_mb", "0.0", False),
    ("Cost", "Training min", "train_min", "0.0", False),
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
                c.number_format = fmt
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

    # Detection against false alarms: the screening trade-off, side by side so
    # a model that detects well by firing constantly is visible as such.
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

    f1 = BarChart()
    f1.type, f1.style, f1.height, f1.width = "col", 10, 9, 22
    f1.title = "F1 at the held-out operating point"
    f1.y_axis.title = "F1"
    c = col_of("f1")
    f1.add_data(Reference(ws_data, min_col=c, min_row=2, max_row=n + 2),
                titles_from_data=True)
    f1.set_categories(names)
    ws.add_chart(f1, "A20")

    # Accuracy against speed. One point per model; the ensemble sits far left.
    sc = ScatterChart()
    sc.style, sc.height, sc.width = 13, 9, 22
    sc.title = "F1 against throughput"
    sc.x_axis.title = "FPS"
    sc.y_axis.title = "F1"
    xs = Reference(ws_data, min_col=col_of("fps"), min_row=3, max_row=n + 2)
    ys = Reference(ws_data, min_col=col_of("f1"), min_row=2, max_row=n + 2)
    s = Series(ys, xs, title_from_data=True)
    s.marker.symbol = "circle"
    s.graphicalProperties.line.noFill = True
    sc.series.append(s)
    ws.add_chart(sc, "A39")


NOTES = [
    ("How to read this workbook", True),
    ("", False),
    ("Operating point", True),
    ("Every rate in the first group is taken at the confidence threshold "
     "chosen on synthetic validation, never at the threshold that maximises "
     "F1 on the test set. Tuning a threshold on the data it is then scored "
     "against is optimistic by construction.", False),
    ("", False),
    ("Why the first group is NOT a fair comparison", True),
    ("Each arm chose its own threshold independently, and those thresholds "
     "land anywhere from 0.155 to 0.500. So each arm's detection and false "
     "alarm rates are read off a different point of its own ROC curve. An arm "
     "whose threshold landed low looks strong on detection and weak on false "
     "alarms; one whose threshold landed high looks the reverse. Comparing "
     "architectures on those columns compares threshold placement, not "
     "architecture.", False),
    ("", False),
    ("Detection at matched false alarms", True),
    ("These are the comparable columns. Every arm is read at the same false "
     "alarm budget, taken as the best detection rate its sweep reaches "
     "without exceeding that budget. Differences here are attributable to the "
     "model. Lead the architecture comparison with these.", False),
    ("", False),
    ("Hit % when fired", True),
    ("Detection % counts any frame where the model fired anywhere in the "
     "image. This column is the fraction of those firings that actually "
     "landed on an annotated object at IoU >= 0.5. It runs 15-30% across "
     "arms, so most frames credited as a detection are not localised on the "
     "e-waste. Detection % is a frame-level alarm rate, not a localisation "
     "rate.", False),
    ("", False),
    ("mIoU, Dice and n matched are not ranked", True),
    ("mIoU is measured at each arm's own threshold, and a higher threshold "
     "mechanically raises it by discarding the loosely-aimed low-confidence "
     "boxes. The arm with the highest mIoU has the fewest matches. Read these "
     "with n matched beside them; they are not a ranking.", False),
    ("", False),
    ("Oracle detection %", True),
    ("The detection rate at the test-set-optimal threshold. It is an upper "
     "bound that assumes the answer is already known, not an achievable "
     "result. It is reported so the size of that gap stays visible.", False),
    ("", False),
    ("Precision, recall and F1", True),
    ("These depend on the 400:747 ratio of positives to negatives in the test "
     "split. That ratio is an artefact of how the split was drawn, not a "
     "real-world rate of contamination. Detection rate and false alarm rate "
     "do not share this dependence and should lead any comparison.", False),
    ("", False),
    ("mAP columns", True),
    ("Measured on synthetic validation composites, not on the real "
     "photographs. They describe box quality on the training distribution.",
     False),
    ("", False),
    ("Annotation set", True),
    ("Localisation is measured against a stratified sample of 100 test "
     "photographs (45 electrical cables, 38 electronic chips, 17 "
     "smartphones), hand-drawn, matching annotations/subset.json. An earlier "
     "pass left 27 extra cable annotations in place, which skewed the sample "
     "to 56.7% cables against a test set that is 45% cables; those were "
     "removed and every arm re-evaluated.", False),
    ("", False),
    ("Early stopping", True),
    ("All arms trained up to 120 epochs, batch 16, seed 0, with early "
     "stopping at patience 30 on validation mAP@50:95. Arms that converged "
     "sooner were stopped by that criterion, so epoch counts differ by "
     "design.", False),
    ("", False),
    ("Ensemble", True),
    ("Weighted box fusion over all 11 members. Member weights come from "
     "held-out synthetic validation F1, never from the test set. Its cost row "
     "is the sum over members: running 11 detectors costs roughly 11 times "
     "the inference of one.", False),
    ("", False),
    ("Highlighting", True),
    ("A shaded cell is the leading value in that column. Differences of "
     "around four points are inside the noise floor of this test set; treat "
     "them as ties.", False),
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
    ap.add_argument("--pool", type=int, default=60)
    args = ap.parse_args()

    mt = load_table_module()
    rows, missing = [], []
    for label, suffix in mt.MODELS:
        row = mt.collect(args.pool, label, suffix)
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
