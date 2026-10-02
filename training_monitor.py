#!/usr/bin/env python3
"""
training_monitor.py — Live progress window for any training command.

Usage:
    py -3.11 training_monitor.py                          # default: setfit_trainer.py --train
    py -3.11 training_monitor.py --cmd "py -3.11 train.py --train"
    py -3.11 training_monitor.py py -3.11 sv_classifier.py --train
"""

import argparse
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import scrolledtext
import queue
import time

POLL_MS  = 500


class TrainingMonitor:
    def __init__(self, root, cmd=None):
        self.root = root
        self.root.title("SetFit Training Monitor")
        self.root.geometry("900x600")
        self.root.configure(bg="#1e1e1e")

        # ── Header ────────────────────────────────────────────────────────
        header = tk.Frame(root, bg="#1e1e1e")
        header.pack(fill=tk.X, padx=10, pady=(10, 0))

        tk.Label(header, text="SetFit Training Monitor",
                 font=("Segoe UI", 14, "bold"), fg="#61afef", bg="#1e1e1e").pack(side=tk.LEFT)

        self.status_label = tk.Label(header, text="● Starting...",
                                     font=("Segoe UI", 10), fg="#e5c07b", bg="#1e1e1e")
        self.status_label.pack(side=tk.RIGHT)

        # ── Phase indicator ───────────────────────────────────────────────
        phase_frame = tk.Frame(root, bg="#1e1e1e")
        phase_frame.pack(fill=tk.X, padx=10, pady=5)

        self.phases = {
            "Loading":       tk.Label(phase_frame, text="① Loading index",    font=("Segoe UI", 9), fg="#4b5263", bg="#1e1e1e", padx=8),
            "Fine-tuning":   tk.Label(phase_frame, text="② Fine-tuning encoder", font=("Segoe UI", 9), fg="#4b5263", bg="#1e1e1e", padx=8),
            "Embedding":     tk.Label(phase_frame, text="③ Building embeddings", font=("Segoe UI", 9), fg="#4b5263", bg="#1e1e1e", padx=8),
            "Classifiers":   tk.Label(phase_frame, text="④ Training classifiers", font=("Segoe UI", 9), fg="#4b5263", bg="#1e1e1e", padx=8),
            "Done":          tk.Label(phase_frame, text="⑤ Complete",          font=("Segoe UI", 9), fg="#4b5263", bg="#1e1e1e", padx=8),
        }
        for lbl in self.phases.values():
            lbl.pack(side=tk.LEFT)

        # ── Progress bar ──────────────────────────────────────────────────
        bar_frame = tk.Frame(root, bg="#1e1e1e")
        bar_frame.pack(fill=tk.X, padx=10, pady=(0, 5))

        self.canvas = tk.Canvas(bar_frame, height=18, bg="#282c34", highlightthickness=0)
        self.canvas.pack(fill=tk.X)
        self.bar_rect  = self.canvas.create_rectangle(0, 0, 0, 18, fill="#61afef", outline="")
        self.bar_label = self.canvas.create_text(5, 9, anchor="w", text="", fill="white",
                                                  font=("Segoe UI", 8))
        self.canvas.bind("<Configure>", self._redraw_bar)
        self._bar_pct = 0.0

        # ── Log output ────────────────────────────────────────────────────
        self.log_box = scrolledtext.ScrolledText(
            root, bg="#282c34", fg="#abb2bf", font=("Consolas", 9),
            insertbackground="white", relief=tk.FLAT, borderwidth=0,
            wrap=tk.WORD,
        )
        self.log_box.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        self.log_box.tag_config("info",  foreground="#abb2bf")
        self.log_box.tag_config("warn",  foreground="#e5c07b")
        self.log_box.tag_config("phase", foreground="#98c379", font=("Consolas", 9, "bold"))
        self.log_box.tag_config("done",  foreground="#61afef", font=("Consolas", 9, "bold"))
        self.log_box.tag_config("error", foreground="#e06c75")

        self.q = queue.Queue()
        self._phase_order = list(self.phases.keys())
        self._current_phase = None
        self._start_time = time.time()
        self._cmd = cmd

        # Start the training process
        self._proc = None
        threading.Thread(target=self._run_training, daemon=True).start()
        self.root.after(POLL_MS, self._poll_queue)

    def _run_training(self):
        try:
            self._proc = subprocess.Popen(
                self._cmd,
                shell=isinstance(self._cmd, str),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            for line in self._proc.stdout:
                self.q.put(line.rstrip())
            self._proc.wait()
            if self._proc.returncode == 0:
                self.q.put("__DONE__")
            else:
                self.q.put(f"__ERROR__ Exit code {self._proc.returncode}")
        except Exception as e:
            self.q.put(f"__ERROR__ {e}")

    def _poll_queue(self):
        try:
            while True:
                line = self.q.get_nowait()
                self._handle_line(line)
        except queue.Empty:
            pass
        self.root.after(POLL_MS, self._poll_queue)

    def _handle_line(self, line):
        if line == "__DONE__":
            self._set_phase("Done")
            self._set_bar(1.0)
            self.status_label.config(text="● Complete", fg="#98c379")
            elapsed = (time.time() - self._start_time) / 60
            self._append(f"\n✓ Training complete in {elapsed:.1f} min", "done")
            return

        if line.startswith("__ERROR__"):
            self.status_label.config(text="● Error", fg="#e06c75")
            self._append(line, "error")
            return

        # Detect phase transitions
        low = line.lower()
        if "loading index" in low:
            self._set_phase("Loading"); self._set_bar(0.05)
        elif "contrastive" in low or "fine-tun" in low or "setfit model" in low:
            self._set_phase("Fine-tuning"); self._set_bar(0.15)
        elif "embedding" in low or "embed" in low:
            self._set_phase("Embedding"); self._set_bar(0.60)
        elif "trained" in low and "classifier" in low:
            self._set_phase("Classifiers"); self._set_bar(0.85)
        elif "classifiers saved" in low or "index updated" in low:
            self._set_phase("Done"); self._set_bar(1.0)

        # Colour by severity
        tag = "info"
        if "warning" in low or "warn" in low:
            tag = "warn"
        elif "error" in low or "traceback" in low:
            tag = "error"
        elif any(k in low for k in ["fine-tun", "embedding", "trained", "complete", "saved"]):
            tag = "phase"

        self._append(line, tag)

    def _append(self, text, tag="info"):
        self.log_box.insert(tk.END, text + "\n", tag)
        self.log_box.see(tk.END)

    def _set_phase(self, phase):
        if phase == self._current_phase:
            return
        self._current_phase = phase
        for name, lbl in self.phases.items():
            idx_cur = self._phase_order.index(phase)
            idx_lbl = self._phase_order.index(name)
            if idx_lbl < idx_cur:
                lbl.config(fg="#98c379")   # done — green
            elif idx_lbl == idx_cur:
                lbl.config(fg="#e5c07b")   # active — yellow
            else:
                lbl.config(fg="#4b5263")   # pending — grey

    def _set_bar(self, pct):
        self._bar_pct = pct
        self._redraw_bar()

    def _redraw_bar(self, event=None):
        w = self.canvas.winfo_width()
        fill_w = int(w * self._bar_pct)
        self.canvas.coords(self.bar_rect, 0, 0, fill_w, 18)
        pct_text = f"{int(self._bar_pct * 100)}%"
        self.canvas.itemconfig(self.bar_label, text=pct_text)


def main():
    parser = argparse.ArgumentParser(description="Training monitor GUI")
    parser.add_argument("--cmd", type=str, default=None,
                        help="Shell command string to run (e.g. 'py -3.11 setfit_trainer.py --train')")
    args, extra = parser.parse_known_args()

    if args.cmd:
        cmd = args.cmd          # run via shell=True
    elif extra:
        cmd = extra             # positional args → list, shell=False
    else:
        cmd = [sys.executable, "setfit_trainer.py", "--train"]  # default

    root = tk.Tk()
    app = TrainingMonitor(root, cmd=cmd)
    root.mainloop()


if __name__ == "__main__":
    main()
