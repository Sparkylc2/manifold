#!/usr/bin/env python3
"""bakes the latex snippets into pngs the engine can load with EquationCache

js drop a file  tools/latex/equations/<name>.tex
containing a LaTeX math snippet

python3 tools/latex/bake.py

m_eq.draw("<name>", x, y, height);

"""

import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- config ----
DPI = 400
COLOR = "black"
FONTSET = "cm"
FORMATS = ["png"]
PAD = 0.05


# matplotlib.rcParams["mathtext.rm"] = "assets/fonts/SpaceGrotesk-Regular.ttf"
# matplotlib.rcParams["mathtext.it"] = "assets/fonts/SpaceGrotesk-Medium.ttf"

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "equations")
OUT = os.path.abspath(os.path.join(HERE, "..", "..", "assets", "equations"))


def render(name, latex):
    expr = latex.strip()
    if not expr.startswith("$"):
        expr = "$" + expr + "$"
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, expr, color=COLOR, fontsize=48)
    for fmt in FORMATS:
        fig.savefig(
            os.path.join(OUT, name + "." + fmt),
            dpi=DPI,
            transparent=True,
            bbox_inches="tight",
            pad_inches=PAD,
        )
        print("  " + name + "." + fmt)
    plt.close(fig)


def main():
    matplotlib.rcParams["mathtext.fontset"] = FONTSET
    os.makedirs(OUT, exist_ok=True)
    files = sorted(glob.glob(os.path.join(SRC, "*.tex")))
    if not files:
        print("no .tex files in " + SRC)
        return
    print("baking " + str(len(files)) + " -> " + OUT)
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        with open(f) as fh:
            render(name, fh.read())


if __name__ == "__main__":
    main()
