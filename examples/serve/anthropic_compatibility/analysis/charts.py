# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dependency-free SVG charts for the HTML report: time-series lines and histograms.

Colours are CSS variables (--series-N, --ink, --grid) so the page stylesheet owns light/dark mode.
Every mark carries a <title> so hovering shows the value; that is the whole interaction layer.
"""
from __future__ import annotations

import html
import math
import statistics
from datetime import datetime, timezone

W, H = 720, 220          # chart box
PAD_L, PAD_R, PAD_T, PAD_B = 56, 16, 28, 34
MAX_POINTS = 300          # per series after time-bucket averaging


def _esc(text) -> str:
    return html.escape(str(text))


def _ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    """Round tick positions covering [lo, hi]."""
    if hi <= lo:
        hi = lo + 1
    raw = (hi - lo) / n
    step = 10 ** math.floor(math.log10(raw))
    for mult in (1, 2, 2.5, 5, 10):
        if raw <= step * mult:
            step *= mult
            break
    first = math.floor(lo / step) * step
    return [first + i * step for i in range(int((hi - first) / step) + 2)]


def _fmt(value: float) -> str:
    if abs(value) >= 1e6:
        return f"{value / 1e6:.1f}M"
    if abs(value) >= 1e4:
        return f"{value / 1e3:.0f}K"
    if abs(value) >= 100:
        return f"{value:.0f}"
    return f"{value:.3g}"


def _clock(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).strftime("%H:%M")


def downsample(points: list[tuple[float, float]], budget: int = MAX_POINTS) -> list[tuple[float, float]]:
    """Average into equal time buckets so a 100k-iteration series stays legible."""
    points = [(t, v) for t, v in points if t is not None and v is not None]
    if len(points) <= budget:
        return sorted(points)
    points.sort()
    t0, t1 = points[0][0], points[-1][0]
    width = (t1 - t0) / budget or 1.0
    buckets: dict[int, list[float]] = {}
    for t, v in points:
        buckets.setdefault(min(budget - 1, int((t - t0) / width)), []).append(v)
    return [(t0 + (i + 0.5) * width, statistics.fmean(vs)) for i, vs in sorted(buckets.items())]


def line_chart(series: list[tuple[str, list[tuple[float, float]]]], title: str, ylabel: str,
               y_max: float | None = None, unit: str = "") -> str:
    """Time on x (UTC clock), one 2px line per series, legend when there are two or more."""
    series = [(name, downsample(pts)) for name, pts in series]
    series = [(name, pts) for name, pts in series if pts]
    if not series:
        return f'<div class="chart empty"><h4>{_esc(title)}</h4><p>no data</p></div>'
    xs = [t for _, pts in series for t, _ in pts]
    ys = [v for _, pts in series for _, v in pts]
    x0, x1 = min(xs), max(xs) if max(xs) > min(xs) else min(xs) + 1
    y_top = y_max if y_max is not None else max(ys) * 1.05 or 1.0
    y_lo = min(0.0, min(ys))
    sx = lambda t: PAD_L + (t - x0) / (x1 - x0) * (W - PAD_L - PAD_R)
    sy = lambda v: PAD_T + (1 - (v - y_lo) / (y_top - y_lo)) * (H - PAD_T - PAD_B)

    parts = [f'<svg viewBox="0 0 {W} {H}" class="viz" role="img" aria-label="{_esc(title)}">']
    for tick in _ticks(y_lo, y_top):
        if y_lo <= tick <= y_top:
            y = sy(tick)
            parts.append(f'<line x1="{PAD_L}" x2="{W - PAD_R}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>'
                         f'<text x="{PAD_L - 6}" y="{y + 4:.1f}" class="tick" text-anchor="end">{_fmt(tick)}</text>')
    for i in range(6):
        t = x0 + (x1 - x0) * i / 5
        parts.append(f'<text x="{sx(t):.1f}" y="{H - 10}" class="tick" text-anchor="middle">{_clock(t)}</text>')
    parts.append(f'<line x1="{PAD_L}" x2="{W - PAD_R}" y1="{sy(y_lo):.1f}" y2="{sy(y_lo):.1f}" class="axis"/>')
    for k, (name, pts) in enumerate(series):
        colour = f"var(--series-{k % 8 + 1})"
        path = " ".join(f"{'M' if i == 0 else 'L'}{sx(t):.1f},{sy(min(v, y_top)):.1f}" for i, (t, v) in enumerate(pts))
        parts.append(f'<path d="{path}" fill="none" stroke="{colour}" stroke-width="2" stroke-linejoin="round"/>')
        for t, v in pts:  # hover targets
            parts.append(f'<circle cx="{sx(t):.1f}" cy="{sy(min(v, y_top)):.1f}" r="5" fill="transparent">'
                         f'<title>{_esc(name)} {_clock(t)} UTC: {_fmt(v)}{unit}</title></circle>')
    parts.append(f'<text x="{PAD_L}" y="16" class="label">{_esc(ylabel)}</text>')
    parts.append("</svg>")
    legend = ""
    if len(series) > 1:
        legend = '<div class="legend">' + "".join(
            f'<span><i style="background:var(--series-{k % 8 + 1})"></i>{_esc(name)}</span>'
            for k, (name, _) in enumerate(series)) + "</div>"
    return f'<div class="chart"><h4>{_esc(title)}</h4>{"".join(parts)}{legend}</div>'


def histogram(values: list[float | None], title: str, xlabel: str, log_x: bool = False, bins: int = 30) -> str:
    """Counts per bin; log-spaced bins for long-tailed quantities. p50 and p90 are drawn as rules."""
    kept = sorted(v for v in values if v is not None and (v > 0 or not log_x))
    if not kept:
        return f'<div class="chart empty"><h4>{_esc(title)}</h4><p>no data</p></div>'
    lo, hi = kept[0], kept[-1]
    if hi <= lo:
        hi = lo + 1
    f = (lambda v: math.log10(v)) if log_x else (lambda v: v)
    edges = [f(lo) + (f(hi) - f(lo)) * i / bins for i in range(bins + 1)]
    counts = [0] * bins
    for v in kept:
        counts[min(bins - 1, int((f(v) - edges[0]) / (edges[-1] - edges[0]) * bins))] += 1
    top = max(counts) * 1.1
    sx = lambda e: PAD_L + (e - edges[0]) / (edges[-1] - edges[0]) * (W - PAD_L - PAD_R)
    sy = lambda c: PAD_T + (1 - c / top) * (H - PAD_T - PAD_B)
    inv = (lambda e: 10 ** e) if log_x else (lambda e: e)

    parts = [f'<svg viewBox="0 0 {W} {H}" class="viz" role="img" aria-label="{_esc(title)}">']
    for tick in _ticks(0, top, 4):
        if 0 <= tick <= top:
            parts.append(f'<line x1="{PAD_L}" x2="{W - PAD_R}" y1="{sy(tick):.1f}" y2="{sy(tick):.1f}" class="grid"/>'
                         f'<text x="{PAD_L - 6}" y="{sy(tick) + 4:.1f}" class="tick" text-anchor="end">{_fmt(tick)}</text>')
    for i, c in enumerate(counts):
        x, w = sx(edges[i]), sx(edges[i + 1]) - sx(edges[i])
        parts.append(f'<rect x="{x + 1:.1f}" y="{sy(c):.1f}" width="{max(w - 2, 1):.1f}" height="{sy(0) - sy(c):.1f}" '
                     f'rx="2" fill="var(--series-1)"><title>{_fmt(inv(edges[i]))} to {_fmt(inv(edges[i + 1]))}: {c}</title></rect>')
    for i in range(6):
        e = edges[0] + (edges[-1] - edges[0]) * i / 5
        parts.append(f'<text x="{sx(e):.1f}" y="{H - 10}" class="tick" text-anchor="middle">{_fmt(inv(e))}</text>')
    for q, cls in ((0.5, "p50"), (0.9, "p90")):
        v = kept[min(len(kept) - 1, int(round(q * (len(kept) - 1))))]
        parts.append(f'<line x1="{sx(f(v)):.1f}" x2="{sx(f(v)):.1f}" y1="{PAD_T}" y2="{sy(0):.1f}" class="{cls}"/>'
                     f'<text x="{sx(f(v)) + 3:.1f}" y="{PAD_T + 10}" class="tick">{cls} {_fmt(v)}</text>')
    parts.append(f'<line x1="{PAD_L}" x2="{W - PAD_R}" y1="{sy(0):.1f}" y2="{sy(0):.1f}" class="axis"/>')
    parts.append(f'<text x="{PAD_L}" y="16" class="label">{_esc(xlabel)}{" (log)" if log_x else ""} · n={len(kept)}</text>')
    parts.append("</svg>")
    return f'<div class="chart"><h4>{_esc(title)}</h4>{"".join(parts)}</div>'


CSS = """
:root { color-scheme: light; --surface: #fcfcfb; --page: #f9f9f7; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --series-1: #2a78d6; --series-2: #eb6834; --series-3: #1baf7a; --series-4: #eda100;
  --series-5: #e87ba4; --series-6: #008300; --series-7: #4a3aa7; --series-8: #e34948; --warn: #c2410c; }
@media (prefers-color-scheme: dark) { :root { color-scheme: dark; --surface: #1a1a19; --page: #0d0d0d; --ink: #ffffff;
  --ink-2: #c3c2b7; --grid: #2c2c2a; --axis: #383835; --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70;
  --series-4: #c98500; --series-5: #d55181; --series-7: #9085e9; --series-8: #e66767; } }
body { font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; color: var(--ink); background: var(--page); margin: 0; padding: 24px 32px; }
h1 { font-size: 22px; margin: 0 0 4px; } h2 { font-size: 18px; margin: 32px 0 8px; border-bottom: 1px solid var(--grid); padding-bottom: 4px; }
h3 { font-size: 15px; margin: 20px 0 6px; } h4 { font-size: 13px; margin: 0 0 4px; color: var(--ink-2); font-weight: 600; }
p.sub, li { color: var(--ink-2); } p.sub { margin: 4px 0 10px; }
table { border-collapse: collapse; margin: 6px 0 14px; font-variant-numeric: tabular-nums; }
th, td { padding: 3px 10px; text-align: right; border-bottom: 1px solid var(--grid); white-space: nowrap; }
th:first-child, td:first-child { text-align: left; } th { color: var(--ink-2); font-weight: 600; }
.chart { background: var(--surface); border: 1px solid var(--grid); border-radius: 6px; padding: 10px 12px 6px; margin: 8px 0; max-width: 760px; }
.grid-2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(380px, 1fr)); gap: 8px; }
.grid-2 .chart { max-width: none; } svg.viz { width: 100%; height: auto; display: block; }
svg .grid { stroke: var(--grid); stroke-width: 1; } svg .axis { stroke: var(--axis); stroke-width: 1; }
svg .tick { fill: var(--muted); font-size: 11px; } svg .label { fill: var(--ink-2); font-size: 12px; }
svg .p50 { stroke: var(--ink-2); stroke-dasharray: 4 3; } svg .p90 { stroke: var(--warn); stroke-dasharray: 4 3; }
.legend { display: flex; flex-wrap: wrap; gap: 4px 14px; font-size: 12px; color: var(--ink-2); margin-top: 4px; }
.legend i { display: inline-block; width: 12px; height: 3px; margin-right: 6px; vertical-align: middle; border-radius: 2px; }
.note { background: var(--surface); border-left: 3px solid var(--series-1); padding: 6px 12px; margin: 8px 0; color: var(--ink-2); }
.warn { border-left-color: var(--warn); } code { font-size: 12.5px; }
"""
