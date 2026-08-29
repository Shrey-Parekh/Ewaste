"""
09_annotate.py
--------------
Hand-draws ground-truth boxes on the held-out e-waste photographs.

This is a one-off manual step, not part of the automated run order, and it is
optional: everything except mIoU and Dice is measured without it.

It exists because ewaste_test carries no boxes. Without them the detection
rate counts a photograph as found when the model fires anywhere in the frame,
which is an upper bound rather than a measurement of localisation -- the model
can be right for the wrong reason and still score. Drawing the boxes turns
that figure into a localisation-verified one, and is what makes mIoU and Dice
computable at all.

The boxes have to be drawn by hand. Pre-filling them from a trained detector
would make the ground truth a function of the model being scored, and the
resulting IoU would measure self-consistency rather than accuracy.

Controls
  drag left        draw a box
  right-click      delete the box under the cursor
  u                undo the last box on this image
  left / right     previous / next photograph, saving as it goes
  q or Escape      save and quit

Progress is the presence of a label file, so an image deliberately left empty
still counts as done and will not be offered again.

Run:    python 09_annotate.py
        python 09_annotate.py --check     coverage report, no window
Output: annotations/ewaste_test/<stem>.txt   YOLO format, one class
"""

from pathlib import Path
import argparse
import csv

from PIL import Image, ImageTk

from pipeline_common import load_image

ROOT = Path(__file__).parent
SPLITS = ROOT / "splits"
OUT = ROOT / "annotations" / "ewaste_test"

MIN_BOX_PX = 4
# 74 of the 400 test photographs are under 400 px on the longest edge and the
# smallest is 189 px. Shown at their own size they are too small to place a box
# on accurately, which would put avoidable error straight into the mIoU this
# annotation exists to measure, so small images are enlarged to fill the window.
MAX_ZOOM = 4.0
BOX_COLOUR = "#00FF00"
LIVE_COLOUR = "#FFCC00"


def read_manifest():
    with open(SPLITS / "ewaste_test.csv", encoding="utf-8") as f:
        return [ROOT / r["path"] for r in csv.DictReader(f)]


def label_path(image_path):
    return OUT / f"{image_path.stem}.txt"


def load_boxes(image_path, size):
    """Read a label file back into pixel xyxy, or [] when none exists."""
    p = label_path(image_path)
    if not p.exists():
        return []
    w, h = size
    boxes = []
    for line in p.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        _, cx, cy, bw, bh = (float(v) for v in parts)
        boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h,
                      (cx + bw / 2) * w, (cy + bh / 2) * h])
    return boxes


def save_boxes(image_path, size, boxes):
    OUT.mkdir(parents=True, exist_ok=True)
    w, h = size
    lines = []
    for x1, y1, x2, y2 in boxes:
        cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
        bw, bh = abs(x2 - x1) / w, abs(y2 - y1) / h
        lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    label_path(image_path).write_text("\n".join(lines), encoding="utf-8")


def count_boxes(image_path):
    p = label_path(image_path)
    if not p.exists():
        return 0
    return sum(1 for line in p.read_text(encoding="utf-8").splitlines()
               if line.strip())


def report(paths):
    done = [p for p in paths if label_path(p).exists()]
    boxes = sum(count_boxes(p) for p in done)
    print(f"annotated {len(done)} of {len(paths)} photographs, {boxes} boxes")
    empty = [p for p in done if count_boxes(p) == 0]
    if empty:
        print(f"  {len(empty)} marked as containing no visible e-waste")
    if len(done) < len(paths):
        print(f"  {len(paths) - len(done)} remaining -> python 09_annotate.py")
    else:
        print("  complete; 06_evaluate.py will now report mIoU and Dice")


class Annotator:
    def __init__(self, root, paths):
        import tkinter as tk

        self.paths, self.root = paths, root
        self.i = self._first_unannotated()
        self.n_done = sum(1 for p in paths if label_path(p).exists())
        self.boxes, self.drag = [], None
        self.photo = self.image = None
        self.scale = 1.0

        root.title("e-waste annotation")
        self.maxw = int(root.winfo_screenwidth() * 0.85)
        self.maxh = int(root.winfo_screenheight() * 0.80)

        self.canvas = tk.Canvas(root, bg="#222222", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.status = tk.Label(root, anchor="w", font=("Segoe UI", 10))
        self.status.pack(fill="x")

        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<Button-3>", self.on_delete)
        root.bind("<Right>", lambda e: self.step(1))
        root.bind("<Left>", lambda e: self.step(-1))
        root.bind("u", lambda e: self.undo())
        root.bind("q", lambda e: self.quit())
        root.bind("<Escape>", lambda e: self.quit())
        root.protocol("WM_DELETE_WINDOW", self.quit)

        self.show()

    def _first_unannotated(self):
        for n, p in enumerate(self.paths):
            if not label_path(p).exists():
                return n
        return 0

    def show(self):
        path = self.paths[self.i]
        self.image = load_image(path)
        w, h = self.image.size
        self.scale = min(self.maxw / w, self.maxh / h, MAX_ZOOM)
        disp = self.image.resize((int(w * self.scale), int(h * self.scale)),
                                 Image.LANCZOS)
        self.photo = ImageTk.PhotoImage(disp)
        self.canvas.config(width=disp.width, height=disp.height)
        self.boxes = load_boxes(path, self.image.size)
        self.redraw()

    def redraw(self):
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)
        for b in self.boxes:
            self.canvas.create_rectangle(*[v * self.scale for v in b],
                                         outline=BOX_COLOUR, width=2)
        self.status.config(
            text=f"  {self.i + 1} / {len(self.paths)}   "
                 f"{len(self.boxes)} box(es)   "
                 f"annotated {self.n_done}/{len(self.paths)}   "
                 f"{self.paths[self.i].name}")

    def on_press(self, e):
        self.drag = (e.x, e.y)

    def on_move(self, e):
        if not self.drag:
            return
        self.redraw()
        self.canvas.create_rectangle(*self.drag, e.x, e.y,
                                     outline=LIVE_COLOUR, width=2)

    def on_release(self, e):
        if not self.drag:
            return
        x0, y0 = self.drag
        self.drag = None
        x1, x2 = sorted((x0, e.x))
        y1, y2 = sorted((y0, e.y))
        if x2 - x1 >= MIN_BOX_PX and y2 - y1 >= MIN_BOX_PX:
            self.boxes.append([v / self.scale for v in (x1, y1, x2, y2)])
        self.redraw()

    def on_delete(self, e):
        x, y = e.x / self.scale, e.y / self.scale
        inside = [b for b in self.boxes
                  if b[0] <= x <= b[2] and b[1] <= y <= b[3]]
        if inside:
            # smallest first, so a box nested inside another stays reachable
            self.boxes.remove(
                min(inside, key=lambda b: (b[2] - b[0]) * (b[3] - b[1])))
            self.redraw()

    def undo(self):
        if self.boxes:
            self.boxes.pop()
            self.redraw()

    def save(self):
        first_time = not label_path(self.paths[self.i]).exists()
        save_boxes(self.paths[self.i], self.image.size, self.boxes)
        if first_time:
            self.n_done += 1

    def step(self, d):
        self.save()
        self.i = (self.i + d) % len(self.paths)
        self.show()

    def quit(self):
        self.save()
        report(self.paths)
        self.root.destroy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="print coverage and exit without opening the window")
    args = ap.parse_args()

    paths = read_manifest()
    missing = [p for p in paths if not p.exists()]
    if missing:
        print(f"[!] {len(missing)} photographs in the manifest are not on disk, "
              f"first: {missing[0]}")
        return

    if args.check:
        report(paths)
        return

    import tkinter as tk

    root = tk.Tk()
    Annotator(root, paths)
    root.mainloop()


if __name__ == "__main__":
    main()
