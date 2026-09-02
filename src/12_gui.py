"""
12_gui.py
---------
A launcher for the training and evaluation runs.

Pick one or more models, choose whether to train, evaluate, or both, and press
Run. Selected jobs are queued and executed one after another, because two
trainings will not fit on an 8 GB card at once.

It runs exactly the commands documented in the README and the handoff, as
subprocesses:

    python src/05_train.py    --pool <n> --model <cfg> [--tag <tag>]
    python src/06_evaluate.py --pool <n>               [--tag <tag>]

Nothing is reimplemented here. The launcher chooses arguments and shows output;
if a run behaves differently from the same command typed by hand, that is a bug
in this file.

Run:  python src/12_gui.py
"""

from pathlib import Path
import queue
import subprocess
import sys
import threading

from lib_arms import ARMS, run_name

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent

POOL_DEFAULT = 60
VERDIGRIS = "#1F3B33"
INK = "#2B3A34"
GREY = "#8B978F"
BG = "#F5F7F4"


def status_of(pool, tag):
    """(trained, evaluated) for one arm, read from what is on disk."""
    name = run_name(pool, tag)
    trained = (ROOT / "runs" / "detect" / name / "weights" / "best.pt").exists()
    suffix = f"_{tag}" if tag else ""
    evaluated = (ROOT / f"eval_pool{pool}{suffix}" / "summary.json").exists()
    return trained, evaluated


class Launcher:
    def __init__(self, root):
        import tkinter as tk
        from tkinter import ttk

        self.tk, self.root = tk, root
        self.proc = None
        self.jobs = []
        self.total = 0
        self.stopping = False
        self.lines = queue.Queue()

        root.title("e-waste detector runs")
        root.configure(bg=BG)
        root.geometry("1000x680")

        head = tk.Frame(root, bg=BG)
        head.pack(fill="x", padx=14, pady=(12, 6))
        tk.Label(head, text="Training and evaluation", bg=BG, fg=VERDIGRIS,
                 font=("Segoe UI", 15, "bold")).pack(side="left")
        tk.Label(head, text="pool", bg=BG, fg=GREY,
                 font=("Segoe UI", 9)).pack(side="left", padx=(18, 4))
        self.pool = tk.StringVar(value=str(POOL_DEFAULT))
        tk.Entry(head, textvariable=self.pool, width=6,
                 font=("Consolas", 10)).pack(side="left")
        tk.Button(head, text="Refresh status", command=self.refresh,
                  font=("Segoe UI", 9)).pack(side="left", padx=10)

        body = tk.Frame(root, bg=BG)
        body.pack(fill="both", expand=True, padx=14)

        cols = ("model", "tag", "trained", "evaluated")
        self.tree = ttk.Treeview(body, columns=cols, show="headings",
                                 selectmode="extended", height=12)
        for c, w, a in (("model", 300, "w"), ("tag", 190, "w"),
                        ("trained", 110, "center"), ("evaluated", 110, "center")):
            self.tree.heading(c, text=c.capitalize())
            self.tree.column(c, width=w, anchor=a)
        self.tree.pack(fill="both", expand=False, pady=(4, 8))

        controls = tk.Frame(root, bg=BG)
        controls.pack(fill="x", padx=14)
        self.do_train = tk.BooleanVar(value=True)
        self.do_eval = tk.BooleanVar(value=True)
        tk.Checkbutton(controls, text="Train", variable=self.do_train, bg=BG,
                       font=("Segoe UI", 9)).pack(side="left")
        tk.Checkbutton(controls, text="Evaluate", variable=self.do_eval, bg=BG,
                       font=("Segoe UI", 9)).pack(side="left", padx=(6, 16))
        tk.Button(controls, text="Select all", command=self.select_all,
                  font=("Segoe UI", 9)).pack(side="left")
        tk.Button(controls, text="Select untrained", command=self.select_untrained,
                  font=("Segoe UI", 9)).pack(side="left", padx=6)
        self.run_btn = tk.Button(controls, text="Run selected", command=self.start,
                                 bg=VERDIGRIS, fg="white",
                                 font=("Segoe UI", 10, "bold"), padx=14)
        self.run_btn.pack(side="right")
        self.stop_btn = tk.Button(controls, text="Stop", command=self.stop,
                                  state="disabled", font=("Segoe UI", 9))
        self.stop_btn.pack(side="right", padx=8)

        self.status = tk.Label(root, text="idle", bg=BG, fg=INK, anchor="w",
                               font=("Segoe UI", 9))
        self.status.pack(fill="x", padx=14, pady=(8, 2))

        logwrap = tk.Frame(root, bg=BG)
        logwrap.pack(fill="both", expand=True, padx=14, pady=(0, 12))
        self.log = tk.Text(logwrap, bg="#121a16", fg="#D6E2DA", wrap="none",
                           font=("Consolas", 9), insertbackground="#D6E2DA")
        bar = tk.Scrollbar(logwrap, command=self.log.yview)
        self.log.configure(yscrollcommand=bar.set)
        bar.pack(side="right", fill="y")
        self.log.pack(side="left", fill="both", expand=True)

        root.protocol("WM_DELETE_WINDOW", self.close)
        self.refresh()
        self.root.after(80, self.drain)

    # ---- table -----------------------------------------------------------
    def pool_value(self):
        try:
            return int(self.pool.get())
        except ValueError:
            return POOL_DEFAULT

    def refresh(self):
        for row in self.tree.get_children():
            self.tree.delete(row)
        pool = self.pool_value()
        for label, _cfg, tag in ARMS:
            trained, evaluated = status_of(pool, tag)
            self.tree.insert("", "end", iid=tag or "_base",
                             values=(label, tag or "(none)",
                                     "yes" if trained else "-",
                                     "yes" if evaluated else "-"))

    def selected_arms(self):
        chosen = set(self.tree.selection())
        return [a for a in ARMS if (a[2] or "_base") in chosen]

    def select_all(self):
        self.tree.selection_set([a[2] or "_base" for a in ARMS])

    def select_untrained(self):
        pool = self.pool_value()
        pending = [a[2] or "_base" for a in ARMS if not status_of(pool, a[2])[0]]
        self.tree.selection_set(pending)

    # ---- log -------------------------------------------------------------
    def write(self, text, replace_last=False):
        self.log.configure(state="normal")
        if replace_last:
            self.log.delete("end-2l linestart", "end-1c")
        self.log.insert("end", text)
        self.log.see("end")
        self.log.configure(state="disabled")

    def drain(self):
        """Move worker output into the widget without blocking the UI thread."""
        try:
            while True:
                kind, payload = self.lines.get_nowait()
                if kind == "line":
                    # Ultralytics redraws progress with a carriage return. Show
                    # the latest state rather than one line per redraw.
                    if "\r" in payload:
                        self.write(payload.replace("\r", "").rstrip() + "\n", True)
                    else:
                        self.write(payload)
                elif kind == "status":
                    self.status.configure(text=payload)
                elif kind == "done":
                    self.finish()
        except queue.Empty:
            pass
        self.root.after(80, self.drain)

    # ---- running ---------------------------------------------------------
    def start(self):
        arms = self.selected_arms()
        if not arms:
            self.write("Select at least one model first.\n")
            return
        if not (self.do_train.get() or self.do_eval.get()):
            self.write("Tick Train, Evaluate, or both.\n")
            return

        pool = self.pool_value()
        self.jobs = []
        for label, cfg, tag in arms:
            tail = ["--tag", tag] if tag else []
            if self.do_train.get():
                self.jobs.append((f"{label} - train",
                                  ["05_train.py", "--pool", str(pool),
                                   "--model", cfg] + tail))
            if self.do_eval.get():
                self.jobs.append((f"{label} - evaluate",
                                  ["06_evaluate.py", "--pool", str(pool)] + tail))

        self.stopping = False
        self.run_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.total = len(self.jobs)
        self.write("\n" + "=" * 70 + "\n")
        self.write(f"Queued {self.total} job(s)\n")
        self.write("=" * 70 + "\n")
        threading.Thread(target=self.worker, daemon=True).start()

    def worker(self):
        for n, (title, argv) in enumerate(self.jobs, 1):
            if self.stopping:
                break
            self.lines.put(("status", f"{n} of {self.total}: {title}"))
            cmd = [sys.executable, str(SRC / argv[0])] + argv[1:]
            shown = " ".join(["python", "src/" + argv[0]] + argv[1:])
            self.lines.put(("line", "\n$ " + shown + "\n"))
            try:
                self.proc = subprocess.Popen(
                    cmd, cwd=str(ROOT), stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, bufsize=1,
                    encoding="utf-8", errors="replace")
            except OSError as exc:
                self.lines.put(("line", f"could not start: {exc}\n"))
                break
            for line in self.proc.stdout:
                self.lines.put(("line", line))
            code = self.proc.wait()
            self.proc = None
            if code != 0 and not self.stopping:
                self.lines.put(("line", f"\n[!] exited with code {code}; "
                                        "stopping the queue\n"))
                break
        self.lines.put(("done", None))

    def finish(self):
        self.run_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.status.configure(text="stopped" if self.stopping else "finished")
        self.write("\nQueue finished.\n")
        self.refresh()

    def stop(self):
        self.stopping = True
        self.status.configure(text="stopping after the current job...")
        if self.proc is not None:
            self.proc.terminate()

    def close(self):
        if self.proc is not None:
            self.stopping = True
            self.proc.terminate()
        self.root.destroy()


def main():
    import tkinter as tk

    root = tk.Tk()
    Launcher(root)
    root.mainloop()


if __name__ == "__main__":
    main()
