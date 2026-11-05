"""Render the README figures (SVG, light + dark) from the numbers in benchmark_results.md.

Not part of the tool. Needs matplotlib:  pip install matplotlib ; python docs/make_figures.py
Update the DATA table below after a new benchmark run, then re-run.

Marks follow the dataviz skill: bars <= 24px thick, 4px rounded data-end and square
baseline, 2px surface gap between neighbours, hairline solid grid, legend for two
series, values only at bar tips, text in ink tokens (never the series colour).
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch, PathPatch  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402

DOCS = Path(__file__).resolve().parent
PREVIEW = Path(sys.argv[1]) if len(sys.argv) > 1 else None  # optional folder for PNG previews
DOCS.mkdir(exist_ok=True)

THEMES = {
    "light": dict(
        surface="#fcfcfb",
        ink="#0b0b0b",
        ink2="#52514e",
        muted="#898781",
        grid="#e1e0d9",
        axis="#c3c2b7",
        series=["#2a78d6", "#eb6834", "#1baf7a", "#eda100"],
        on_fill=["#ffffff", "#ffffff", "#ffffff", "#0b0b0b"],
    ),
    "dark": dict(
        surface="#1a1a19",
        ink="#f0efec",
        ink2="#c3c2b7",
        muted="#898781",
        grid="#2c2c2a",
        axis="#383835",
        series=["#3987e5", "#d95926", "#199e70", "#c98500"],
        on_fill=["#ffffff", "#ffffff", "#ffffff", "#0b0b0b"],
    ),
}
matplotlib.rcParams["svg.fonttype"] = "none"  # keep text as text, in the reader's system font
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Segoe UI", "Helvetica", "Arial", "DejaVu Sans"]

# --- the measured numbers (benchmark_results.md, 2026-11-05 03:30), ordered by document size ---
SAMPLES = [
    ("injection_attempt", 299),
    ("meeting_notes", 317),
    ("email_thread", 838),
    ("transcript", 2801),
]
# model -> sample -> (decode tok/s, sd), (total s, sd), (peak RAM GB, sd)
DATA = {
    "llama3.2:1b": {
        "injection_attempt": ((22.3, 0.2), (13.0, 0.5), (1.65, 0.00)),
        "meeting_notes": ((20.6, 0.8), (12.5, 0.3), (1.65, 0.00)),
        "email_thread": ((18.9, 0.0), (27.5, 0.3), (1.65, 0.00)),
        "transcript": ((17.0, 0.1), (150.9, 1.1), (1.89, 0.02)),
    },
    "llama3.2:3b": {
        "injection_attempt": ((11.5, 0.6), (26.1, 0.7), (2.70, 0.00)),
        "meeting_notes": ((11.8, 1.3), (38.4, 4.9), (2.70, 0.00)),
        "email_thread": ((9.2, 0.1), (76.8, 1.4), (2.70, 0.00)),
        "transcript": ((9.2, 0.4), (233.1, 10.5), (3.48, 0.09)),
    },
}
MODELS = list(DATA)


def px_to_data(ax, px_x=0.0, px_y=0.0):
    """How many data units a number of pixels is, on each axis of `ax`."""
    bbox = ax.get_window_extent()
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    return px_x * (x1 - x0) / bbox.width, px_y * (y1 - y0) / bbox.height


def rounded_hbar(ax, x0, x1, yc, height_px, color, round_left=False):
    """A horizontal bar from x0 to x1 centred on yc: 4px rounded data-end, square baseline."""
    rx, ry = px_to_data(ax, 4, 4)
    _, h = px_to_data(ax, 0, height_px)
    yb, yt = yc - h / 2, yc + h / 2
    rx = min(rx, (x1 - x0) / 2)
    verts, codes = [], []

    def line(p):
        verts.append(p)
        codes.append(MPath.LINETO)

    def corner(ctrl, end):
        verts.extend([ctrl, end])
        codes.extend([MPath.CURVE3, MPath.CURVE3])

    if round_left:
        verts.append((x0 + rx, yb))
        codes.append(MPath.MOVETO)
    else:
        verts.append((x0, yb))
        codes.append(MPath.MOVETO)
    line((x1 - rx, yb))
    corner((x1, yb), (x1, yb + ry))
    line((x1, yt - ry))
    corner((x1, yt), (x1 - rx, yt))
    if round_left:
        line((x0 + rx, yt))
        corner((x0, yt), (x0, yt - ry))
        line((x0, yb + ry))
        corner((x0, yb), (x0 + rx, yb))
    else:
        line((x0, yt))
        line((x0, yb))
    codes.append(MPath.CLOSEPOLY)
    verts.append(verts[0])
    ax.add_patch(PathPatch(MPath(verts, codes), facecolor=color, edgecolor="none", zorder=3))


def style_axes(ax, t, xlabel, xmax, ticks):
    ax.set_facecolor(t["surface"])
    ax.set_xlim(0, xmax)
    ax.set_xticks(ticks)
    ax.tick_params(axis="x", colors=t["muted"], labelsize=9, length=0, pad=4)
    ax.tick_params(axis="y", colors=t["ink2"], labelsize=9.5, length=0, pad=6)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["axis"])
    ax.spines["bottom"].set_linewidth(0.8)
    ax.grid(axis="x", color=t["grid"], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(xlabel, loc="left", color=t["ink"], fontsize=11, pad=10)


def benchmark_figure(mode):
    t = THEMES[mode]
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.9), dpi=100, facecolor=t["surface"])
    fig.subplots_adjust(left=0.19, right=0.985, top=0.70, bottom=0.14, wspace=0.42)
    panels = [
        ("Decode speed (tokens/s)", 0, 25, [0, 5, 10, 15, 20, 25], "{:.1f}"),
        ("Total time (s)", 1, 260, [0, 60, 120, 180, 240], "{:.0f}"),
        ("Peak RAM (GB)", 2, 4, [0, 1, 2, 3, 4], "{:.2f}"),
    ]
    ylabels = [f"{name}  ·  {words:,} words" for name, words in SAMPLES]
    for ax, (title, idx, xmax, ticks, fmt) in zip(axes, panels, strict=True):
        style_axes(ax, t, title, xmax, ticks)
        ax.set_ylim(-0.6, len(SAMPLES) - 0.4)
        ax.set_yticks(range(len(SAMPLES)))
        ax.set_yticklabels(ylabels if ax is axes[0] else [""] * len(SAMPLES))
        ax.invert_yaxis()
        fig.canvas.draw()  # so pixel->data conversion is right
        _, offset = px_to_data(ax, 0, 13)  # bars 22px thick, 2px apart: centres 13px above/below the row
        for s, model in enumerate(MODELS):
            for row, (name, _) in enumerate(SAMPLES):
                value, sd = DATA[model][name][idx]
                yc = row + (offset if s == 1 else -offset) * -1  # 1b above 3b, like the legend
                rounded_hbar(ax, 0, value, yc, 22, t["series"][s])
                if sd > 0:
                    ax.errorbar(
                        value, yc, xerr=sd, fmt="none", ecolor=t["ink2"], elinewidth=1, capsize=2, zorder=4
                    )
                pad_x, _ = px_to_data(ax, 6 + (8 if sd > 0 else 0), 0)
                ax.text(
                    value + pad_x,
                    yc,
                    fmt.format(value),
                    va="center",
                    ha="left",
                    color=t["ink2"],
                    fontsize=8.5,
                    zorder=5,
                )
    fig.text(
        0.19,
        0.93,
        "llama3.2:1b vs llama3.2:3b on the four sample documents",
        color=t["ink"],
        fontsize=13,
        weight="semibold",
        ha="left",
    )
    fig.text(
        0.19,
        0.86,
        "CPU only (Intel i5-11320H, 16 GB), Ollama 0.40.14, 3 runs per bar, mean with ± sd whiskers. Measured 2026-11-05.",
        color=t["muted"],
        fontsize=9,
        ha="left",
    )
    handles = [Patch(facecolor=t["series"][i], edgecolor="none", label=m) for i, m in enumerate(MODELS)]
    leg = fig.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.985),
        ncol=2,
        frameon=False,
        fontsize=9.5,
        handlelength=1.2,
        handleheight=0.9,
        columnspacing=1.4,
    )
    for text in leg.get_texts():
        text.set_color(t["ink2"])
    return fig


def budget_figure(mode):
    t = THEMES[mode]
    parts = [
        ("Prompt: system + template + retry", 400),
        ("Document chunk", 2000),
        ("Spare, for the token estimate", 672),
        ("Reserved for the answer", 1024),
    ]
    fig, ax = plt.subplots(figsize=(10.4, 1.55), dpi=100, facecolor=t["surface"])
    fig.subplots_adjust(left=0.02, right=0.98, top=0.72, bottom=0.36)
    ax.set_facecolor(t["surface"])
    ax.set_xlim(0, 4096)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.canvas.draw()
    gap, _ = px_to_data(ax, 2, 0)
    x = 0
    for i, (label, size) in enumerate(parts):
        x0, x1 = x + (gap / 2 if i else 0), x + size - (gap / 2 if i < len(parts) - 1 else 0)
        rounded_hbar(ax, x0, x1, 0.5, 28, t["series"][i], round_left=True)
        text = f"{size:,}" if size < 500 else f"{label.split(':')[0].split(',')[0]}  {size:,}"
        pad_x, _ = px_to_data(ax, 8, 0)
        ax.text(
            x0 + pad_x,
            0.5,
            text,
            va="center",
            ha="left",
            color=t["on_fill"][i],
            fontsize=9.5,
            weight="semibold",
            zorder=5,
        )
        x += size
    fig.text(
        0.02,
        0.86,
        "One request must fit the 4,096-token context window (num_ctx)",
        color=t["ink"],
        fontsize=12.5,
        weight="semibold",
        ha="left",
    )
    handles = [
        Patch(facecolor=t["series"][i], edgecolor="none", label=label) for i, (label, _) in enumerate(parts)
    ]
    leg = fig.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.015, 0.0),
        ncol=4,
        frameon=False,
        fontsize=9,
        handlelength=1.1,
        handleheight=0.9,
        columnspacing=1.6,
    )
    for text in leg.get_texts():
        text.set_color(t["ink2"])
    return fig


for mode in THEMES:
    for name, maker in [("benchmark", benchmark_figure), ("context_budget", budget_figure)]:
        fig = maker(mode)
        fig.savefig(DOCS / f"{name}-{mode}.svg", format="svg", facecolor=fig.get_facecolor())
        if PREVIEW:
            fig.savefig(PREVIEW / f"{name}-{mode}.png", format="png", dpi=130, facecolor=fig.get_facecolor())
        plt.close(fig)
        print("wrote", DOCS / f"{name}-{mode}.svg")
