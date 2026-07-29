#!/usr/bin/env python3

import argparse
import builtins
import os
from pathlib import Path
import runpy
import sys

OUTPUT_EXTS = {".png", ".jpg", ".jpeg", ".pdf", ".csv", ".txt", ".json"}


def is_write_mode(mode):
    return any(x in mode for x in ("w", "a", "x", "+"))


def redirected_path(path, output_dir):
    p = Path(path)

    if p.suffix.lower() not in OUTPUT_EXTS:
        return path

    out = output_dir / p.name
    print(f"[redirect] {p} -> {out}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("script", help="Python script to run")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workdir", default=None)
    args = parser.parse_args()

    script = Path(args.script).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.workdir:
        workdir = Path(args.workdir).resolve()
    else:
        workdir = script.parent

    print(f"[runner] script     = {script}")
    print(f"[runner] workdir    = {workdir}")
    print(f"[runner] output_dir = {output_dir}")

    os.environ["SCORECARD_SYSTEM_OUTPUT_DIR"] = str(output_dir)

    # Keep both script folder and workdir importable.
    sys.path.insert(0, str(script.parent))
    sys.path.insert(0, str(workdir))

    # Change cwd so scripts that read relative inputs still work.
    os.chdir(workdir)

    # ------------------------------------------------------------------
    # Redirect builtins.open for write-mode csv/txt/json outputs.
    # ------------------------------------------------------------------
    original_open = builtins.open

    def open_redirect(file, mode="r", *a, **kw):
        if is_write_mode(mode):
            file = redirected_path(file, output_dir)
        return original_open(file, mode, *a, **kw)

    builtins.open = open_redirect

    # ------------------------------------------------------------------
    # Redirect pathlib.Path.open for write-mode csv/txt/json outputs.
    # ------------------------------------------------------------------
    original_path_open = Path.open

    def path_open_redirect(self, mode="r", *a, **kw):
        path = self
        if is_write_mode(mode):
            path = redirected_path(self, output_dir)
        return original_path_open(Path(path), mode, *a, **kw)

    Path.open = path_open_redirect

    # ------------------------------------------------------------------
    # Redirect matplotlib savefig.
    # ------------------------------------------------------------------
    try:
        import matplotlib.figure
        import matplotlib.pyplot as plt

        original_fig_savefig = matplotlib.figure.Figure.savefig
        original_plt_savefig = plt.savefig

        def fig_savefig_redirect(self, fname, *a, **kw):
            fname = redirected_path(fname, output_dir)
            return original_fig_savefig(self, fname, *a, **kw)

        def plt_savefig_redirect(fname, *a, **kw):
            fname = redirected_path(fname, output_dir)
            return original_plt_savefig(fname, *a, **kw)

        matplotlib.figure.Figure.savefig = fig_savefig_redirect
        plt.savefig = plt_savefig_redirect

    except Exception as e:
        print(f"[runner] matplotlib redirect unavailable: {e}")

    # ------------------------------------------------------------------
    # Redirect pandas DataFrame.to_csv.
    # ------------------------------------------------------------------
    try:
        import pandas as pd

        original_to_csv = pd.DataFrame.to_csv

        def to_csv_redirect(self, path_or_buf=None, *a, **kw):
            if path_or_buf is not None:
                path_or_buf = redirected_path(path_or_buf, output_dir)
            return original_to_csv(self, path_or_buf, *a, **kw)

        pd.DataFrame.to_csv = to_csv_redirect

    except Exception as e:
        print(f"[runner] pandas redirect unavailable: {e}")

    # ------------------------------------------------------------------
    # Redirect numpy.savetxt.
    # ------------------------------------------------------------------
    try:
        import numpy as np

        original_savetxt = np.savetxt

        def savetxt_redirect(fname, *a, **kw):
            fname = redirected_path(fname, output_dir)
            return original_savetxt(fname, *a, **kw)

        np.savetxt = savetxt_redirect

    except Exception as e:
        print(f"[runner] numpy redirect unavailable: {e}")

    # Run the script.
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
