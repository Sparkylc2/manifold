#!/usr/bin/env python3
"""bakes the latex snippets into pngs the engine can load with EquationCache

drop a file  tools/latex/equations/<name>.tex
containing a LaTeX math snippet

python3 tools/latex/bake.py

m_eq.draw("<name>", x, y, height);

glyphs are rendered white so the engine can tint them; text.color is set per
draw. font: SF Pro Text if found on this machine (macOS), else Space Grotesk.
"""

import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

# ---- config ----
DPI = 400
COLOR = "white"  # tinted by the engine; white composites against any theme
FORMATS = ["png"]
PAD = 0.06
STROKE = 1.6  # glyph outline (pt) — thickens the math independent of weight

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "equations")
OUT = os.path.abspath(os.path.join(HERE, "..", "..", "assets", "equations"))
FALLBACK = os.path.abspath(
    os.path.join(HERE, "..", "..", "assets", "fonts", "SpaceGrotesk-SemiBold.ttf")
)

# SF Pro Text on macOS, bold weights first; first hit wins
SF_CANDIDATES = [
    "/System/Library/Fonts/SF-Pro-Text-Semibold.otf",
    os.path.expanduser("~/Library/Fonts/SF-Pro-Text-Semibold.otf"),
    "/Library/Fonts/SF-Pro-Text-Semibold.otf",
    "/System/Library/Fonts/SFNSText.ttf",
    "/System/Library/Fonts/SFNS.ttf",
]


def pick_font():
    for p in SF_CANDIDATES + [FALLBACK]:
        if os.path.exists(p):
            return p
    return None


def setup_font():
    path = pick_font()
    if not path:
        print("no font found; using matplotlib cm")
        matplotlib.rcParams["mathtext.fontset"] = "cm"
        return "cm"
    fm.fontManager.addfont(path)
    name = fm.FontProperties(fname=path).get_name()
    matplotlib.rcParams["mathtext.fontset"] = "custom"
    for k in ("rm", "it", "bf", "sf", "cal"):
        matplotlib.rcParams["mathtext.%s" % k] = name
    print("font: " + name + "  (" + path + ")")
    return name


def render(name, latex):
    expr = latex.strip()
    if not expr.startswith("$"):
        expr = "$" + expr + "$"
    fig = plt.figure(figsize=(0.01, 0.01))
    t = fig.text(0, 0, expr, color=COLOR, fontsize=48)
    t.set_path_effects([pe.withStroke(linewidth=STROKE, foreground=COLOR)])
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
    setup_font()
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
