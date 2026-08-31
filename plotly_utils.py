import random
import re
import os
import sys
from tabulate import tabulate
import numpy as np
import pandas as pd
import einops
from einops import einsum
import IPython
from tqdm import tqdm, trange
import functools
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import torch as t
from torch import Tensor

purple = '\x1b[38;2;255;0;255m'
blue = '\x1b[38;2;0;0;255m'
brown = '\x1b[38;2;128;128;0m'
cyan = '\x1b[38;2;0;255;255m'
lime = '\x1b[38;2;0;255;0m'
yellow = '\x1b[38;2;255;255;0m'
red = '\x1b[38;2;255;0;0m'
pink = '\x1b[38;2;255;51;204m'
orange = '\x1b[38;2;255;51;0m'
green = '\x1b[38;2;5;170;20m'
gray = '\x1b[38;2;127;127;127m'
magenta = '\x1b[38;2;128;0;128m'
white = '\x1b[38;2;255;255;255m'
bold = '\033[1m'
underline = '\033[4m'
endc = '\033[0m'

def top_toks_table(logits: Tensor, tokenizer, k: int = 25, show_negative: bool = False, show_probs: bool = True, title: str | None = None):
    assert logits.squeeze().ndim == 1, f"expected the logits for a single token, got shape {tuple(logits.shape)}"
    logits = logits.flatten()
    probs = logits.softmax(-1)
    top = logits.topk(k)
    top_strs = [tokenizer.decode([tok]) for tok in top.indices.tolist()]
    top_vals = top.values.tolist()
    headers = ["Tok", "Value"] + (["Prob"] if show_probs else [])
    cols = [[repr(s) for s in top_strs], top_vals] + ([probs[top.indices].tolist()] if show_probs else [])
    if show_negative:
        bot = logits.topk(k, largest=False)
        bot_strs = [tokenizer.decode([tok]) for tok in bot.indices.tolist()]
        bot_vals = bot.values.tolist()
        headers = [f"Top {h}" for h in headers] + [f"Bot {h}" for h in headers]
        cols += [[repr(s) for s in bot_strs], bot_vals] + ([probs[bot.indices].tolist()] if show_probs else [])
    data = [(i, *(col[i] for col in cols)) for i in range(k)]
    table_str = tabulate(data, headers=["Idx"] + headers, tablefmt="rounded_outline")
    if title is not None:
        lines = table_str.splitlines()
        inner = len(lines[0]) - 2
        print(f"╭{'─' * inner}╮")
        print(f"│{bold}{title.center(inner)}{endc}│")
        print(f"├{'─' * inner}┤")
        print("\n".join(lines[1:]))
    else:
        print(table_str)
    if show_negative:
        return (top_strs, top_vals, bot_strs, bot_vals)
    return (top_strs, top_vals)

# GENERIC PLOTTING FUNCTIONS

def to_numpy(tensor):
    """
    Helper function to convert a tensor to a numpy array. Also works on lists, tuples, and numpy arrays.
    """
    if isinstance(tensor, np.ndarray):
        return tensor
    elif isinstance(tensor, (list, tuple)):
        array = np.array(tensor)
        return array
    elif isinstance(tensor, (t.Tensor, t.nn.parameter.Parameter)):
        tensor = tensor.detach().cpu()
        return tensor.float().numpy() if tensor.dtype == t.bfloat16 else tensor.numpy()  # numpy has no bfloat16
    elif isinstance(tensor, (int, float, bool, str)):
        return np.array(tensor)
    else:
        raise ValueError(f"Input to to_numpy has invalid type: {type(tensor)}")

def to_series(values):
    """
    Normalize plot data to a 1D array, or to a list of 1D arrays if it holds several series.

    Plotly express renames the series it's handed in place, so it needs a list of them rather than a 2D array.
    """
    arr = [to_numpy(series) for series in values] if isinstance(values, list) and not isinstance(values[0], (int, float)) else to_numpy(values)
    return list(arr) if isinstance(arr, np.ndarray) and arr.ndim == 2 else arr


def drop_none(**kwargs) -> dict:
    """
    Drop unset (None) args, so that plotly's own defaults apply to them.
    """
    return {k: v for k, v in kwargs.items() if v is not None}


def margin_dict(margin: int | dict | None) -> dict:
    """
    Layout kwargs for a margin, where an int means the same padding on all four sides.
    """
    if margin is None:
        return {}
    return {"margin": dict.fromkeys("tblr", margin) if isinstance(margin, int) else margin}


def imshow(
    tensor: t.Tensor,
    renderer=None,
    *,
    x: list | None = None,
    y: list | None = None,
    labels: dict | None = None,
    title: str | None = None,
    template: str | None = None,
    size: tuple[int, int] | None = None,
    height: int | None = None,
    width: int | None = None,
    aspect: str | None = None,
    origin: str | None = None,
    zmin: float | None = None,
    zmax: float | None = None,
    range_color: tuple[float, float] | None = None,
    color_continuous_scale: str = "RdBu",
    color_continuous_midpoint: float | None = 0.0,
    facet_col: int | None = None,
    facet_col_wrap: int | None = None,
    animation_frame: int | None = None,
    facet_labels: list[str] | None = None,
    text: list | None = None,
    border: bool = False,
    xaxis_tickangle: float | None = None,
    margin: int | dict | None = None,
    static: bool = False,
    return_fig: bool = False,
    **layout_kwargs,
):
    """
    Heatmap of a 2D (or 3D, faceted) tensor. Extra keyword args go to fig.update_layout.

    Args:
        tensor: 2D array to plot, or 3D for one facet per leading index.
        renderer: plotly renderer to show with, e.g. "browser". None uses the default.
        x, y: tick labels for the columns / rows.
        labels: axis and colorbar names, e.g. {"x": "Head", "y": "Layer", "color": "logit diff"}.
        title: figure title.
        template: plotly theme, e.g. "simple_white".
        size: (height, width) in pixels, shorthand for passing both.
        height, width: figure size in pixels.
        aspect: "equal" for square cells, "auto" to fill the figure.
        origin: "upper" (default) puts row 0 at the top, "lower" at the bottom.
        zmin, zmax: values at the ends of the color scale. Default is the data range.
        range_color: (low, high), another way to set the color scale limits.
        color_continuous_scale: named colorscale. Diverging red-blue by default.
        color_continuous_midpoint: value mapped to the middle color. None for a non-diverging scale.
        facet_col: axis to facet over. A 3D tensor facets over axis 0 by default.
        facet_col_wrap: max facets per row before wrapping to the next row.
        animation_frame: axis to turn into an animation slider instead of facets.
        facet_labels: titles for the facets, in order.
        text: strings to write in the cells: list[list[str]] for 2D, or one of those per facet.
        border: draw a black box around the plot area.
        xaxis_tickangle: rotation of the x tick labels in degrees, applied to every facet.
        margin: padding in pixels. An int sets all four sides.
        static: show as a non-interactive image.
        return_fig: return the figure instead of showing it.
    """
    arr = to_numpy(tensor)
    if size is not None:
        height, width = size
    if arr.ndim == 3 and animation_frame is None and facet_col is None:  # otherwise px would read the 3rd axis as color channels
        facet_col = 0
    px_kwargs = drop_none(
        x=x, y=y, labels=labels, title=title, template=template, height=height, width=width,
        aspect=aspect, origin=origin, zmin=zmin, zmax=zmax, range_color=range_color,
        color_continuous_scale=color_continuous_scale, color_continuous_midpoint=color_continuous_midpoint,
        facet_col=facet_col, facet_col_wrap=facet_col_wrap, animation_frame=animation_frame,
    )
    fig = px.imshow(arr, **px_kwargs).update_layout(**layout_kwargs, **margin_dict(margin))
    if facet_labels:
        set_facet_labels(fig, facet_labels, facet_col_wrap)
    if border:
        fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
        fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    if text:
        if arr.ndim == 2:
            # if 2D, then we assume text is a list of lists of strings
            assert isinstance(text[0], list)
            assert isinstance(text[0][0], str)
            text = [text]
        else:
            # if 3D, then text is either repeated for each facet, or different
            assert isinstance(text[0], list)
            if isinstance(text[0][0], str):
                text = [text for _ in range(len(fig.data))]
        for i, _text in enumerate(text):
            fig.data[i].update(
                text=_text, 
                texttemplate="%{text}", 
                textfont={"size": 12}
            )
    if xaxis_tickangle is not None:  # update_xaxes hits every facet, unlike update_layout(xaxis_...)
        fig.update_xaxes(tickangle=xaxis_tickangle)
    return fig if return_fig else fig.show(renderer=renderer, config={"staticPlot": static})


def reorder_list_in_plotly_way(L: list, col_wrap: int):
    '''
    Helper function, because Plotly orders figures in an annoying way when there's column wrap.
    '''
    L_new = []
    while len(L) > 0:
        L_new.extend(L[-col_wrap:])
        L = L[:-col_wrap]
    return L_new


def set_facet_labels(fig, facet_labels: list[str], facet_col_wrap: int | None):
    """
    Rename the facet titles, undoing plotly's bottom-row-first ordering when the facets wrap.
    """
    assert len(facet_labels) <= len(fig.layout.annotations), f"got {len(facet_labels)} facet_labels but the figure has {len(fig.layout.annotations)} facet titles"
    if facet_col_wrap is not None:
        facet_labels = reorder_list_in_plotly_way(facet_labels, facet_col_wrap)
    for i, label in enumerate(facet_labels):
        fig.layout.annotations[i]['text'] = label


def line(
    y: t.Tensor | list[t.Tensor],
    renderer=None,
    *,
    x=None,
    names: list[str] | None = None,
    labels: dict | None = None,
    title: str | None = None,
    template: str | None = None,
    size: tuple[int, int] | None = None,
    height: int | None = None,
    width: int | None = None,
    color=None,
    markers: bool = False,
    log_y: bool = False,
    hover_name=None,
    xaxis_tickvals: list | None = None,
    use_secondary_yaxis: bool = False,
    hovermode: str = "closest",
    margin: int | dict | None = None,
    return_fig: bool = False,
    **layout_kwargs,
):
    """
    Line plot of one series, or of several if `y` is a list of them. Extra keyword args go to fig.update_layout.

    Args:
        y: 1D values to plot, or several series to plot as separate lines: either a list of 1D series or a 2D array with one line per row.
        renderer: plotly renderer to show with, e.g. "browser". None uses the default.
        x: x values for the points, shared by every line. Defaults to 0, 1, 2, ... With use_secondary_yaxis, a pair of x arrays.
        names: legend name per line, in order. Consumed as the traces are named, so pass a list you don't need afterwards.
        labels: axis names, e.g. {"x": "Layer", "y": "Loss"}. With use_secondary_yaxis, keys are "x", "y1", "y2".
        title: figure title.
        template: plotly theme, e.g. "simple_white".
        size: (height, width) in pixels, shorthand for passing both.
        height, width: figure size in pixels.
        color: values to color the lines by (one line per distinct value).
        markers: draw a marker at each point.
        log_y: log-scale the y axis.
        hover_name: label shown in bold in each point's hover box.
        xaxis_tickvals: tick labels for the x axis, placed at `x` (or at 0, 1, 2, ...).
        use_secondary_yaxis: plot y[0] against a left axis and y[1] against a right one, for series of different scales.
        hovermode: plotly hover behaviour. Defaults to one shared box for all lines at that x.
        margin: padding in pixels. An int sets all four sides.
        return_fig: return the figure instead of showing it.
    """
    if size is not None:
        height, width = size
    layout = {"hovermode": hovermode, **layout_kwargs, **margin_dict(margin)}
    if xaxis_tickvals is not None:
        layout["xaxis"] = dict(
            tickmode = "array",
            tickvals = x if x is not None else np.arange(len(xaxis_tickvals)),
            ticktext = xaxis_tickvals
        )
    if use_secondary_yaxis:
        if labels is not None:
            layout["yaxis_title_text"] = labels.get("y1", None)
            layout["yaxis2_title_text"] = labels.get("y2", None)
            layout["xaxis_title_text"] = labels.get("x", None)
        layout.update(drop_none(title=title, template=template, width=width, height=height))
        fig = make_subplots(specs=[[{"secondary_y": True}]]).update_layout(**layout)
        y0 = to_numpy(y[0])
        y1 = to_numpy(y[1])
        x0, x1 = x if x is not None else [np.arange(len(y0)), np.arange(len(y1))]
        name0, name1 = names if names is not None else ["yaxis1", "yaxis2"]
        fig.add_trace(go.Scatter(y=y0, x=x0, name=name0), secondary_y=False)
        fig.add_trace(go.Scatter(y=y1, x=x1, name=name1), secondary_y=True)
    else:
        y = to_series(y)
        px_kwargs = drop_none(
            x=x, labels=labels, title=title, template=template, height=height, width=width,
            color=color, markers=markers, log_y=log_y, hover_name=hover_name,
        )
        fig = px.line(y=y, **px_kwargs).update_layout(**layout)
        if names is not None:
            fig.for_each_trace(lambda trace: trace.update(name=names.pop(0)))
    return fig if return_fig else fig.show(renderer=renderer)


def scatter(
    x,
    y,
    renderer=None,
    *,
    labels: dict | None = None,
    title: str | None = None,
    template: str | None = None,
    size: tuple[int, int] | None = None,
    height: int | None = None,
    width: int | None = None,
    color=None,
    hover_name=None,
    text=None,
    opacity: float | None = None,
    trendline: str | None = None,
    facet_col=None,
    facet_col_wrap: int | None = None,
    facet_labels: list[str] | None = None,
    add_line: str | None = None,
    textposition: str | None = None,
    margin: int | dict | None = None,
    return_fig: bool = False,
    **layout_kwargs,
):
    """
    Scatter plot of y against x. Extra keyword args go to fig.update_layout.

    Args:
        x, y: point coordinates.
        renderer: plotly renderer to show with, e.g. "browser". None uses the default.
        labels: axis and legend names, e.g. {"x": "Open-proportion", "y": "Contribution"}.
        title: figure title.
        template: plotly theme, e.g. "simple_white".
        size: (height, width) in pixels, shorthand for passing both.
        height, width: figure size in pixels.
        color: values to color the points by.
        hover_name: label shown in bold in each point's hover box.
        text: labels drawn next to each point.
        opacity: point opacity, 0 to 1. Useful when points overlap.
        trendline: fit to overlay, e.g. "ols" or "lowess".
        facet_col: values to split the plot into one subplot per distinct value.
        facet_col_wrap: max facets per row before wrapping to the next row.
        facet_labels: titles for the facets, in order.
        add_line: reference line to draw, one of "y=x", "x=c" or "y=c" for a float c.
        textposition: where `text` sits relative to its point, e.g. "top center".
        margin: padding in pixels. An int sets all four sides.
        return_fig: return the figure instead of showing it.
    """
    x = to_numpy(x)
    y = to_numpy(y)
    if size is not None:
        height, width = size
    px_kwargs = drop_none(
        labels=labels, title=title, template=template, height=height, width=width, color=color,
        hover_name=hover_name, text=text, opacity=opacity, trendline=trendline,
        facet_col=facet_col, facet_col_wrap=facet_col_wrap,
    )
    fig = px.scatter(y=y, x=x, **px_kwargs).update_layout(**layout_kwargs, **margin_dict(margin))
    if add_line is not None:
        xrange = fig.layout.xaxis.range or [x.min(), x.max()]
        yrange = fig.layout.yaxis.range or [y.min(), y.max()]
        add_line = add_line.replace(" ", "")
        if add_line in ["x=y", "y=x"]:
            fig.add_trace(go.Scatter(mode='lines', x=xrange, y=xrange, showlegend=False))
        elif re.match("(x|y)=", add_line):
            try: c = float(add_line.split("=")[1])
            except: raise ValueError(f"Unrecognized add_line: {add_line}. Please use either 'x=y' or 'x=c' or 'y=c' for some float c.")
            x, y = ([c, c], yrange) if add_line[0] == "x" else (xrange, [c, c])
            fig.add_trace(go.Scatter(mode='lines', x=x, y=y, showlegend=False))
        else:
            raise ValueError(f"Unrecognized add_line: {add_line}. Please use either 'x=y' or 'x=c' or 'y=c' for some float c.")
    if facet_labels:
        set_facet_labels(fig, facet_labels, facet_col_wrap)
    if textposition is not None:
        fig.update_traces(textposition=textposition)
    return fig if return_fig else fig.show(renderer=renderer)


def bar(
    tensor,
    renderer=None,
    *,
    x=None,
    names: list[str] | None = None,
    labels: dict | None = None,
    title: str | None = None,
    template: str | None = None,
    size: tuple[int, int] | None = None,
    height: int | None = None,
    width: int | None = None,
    color=None,
    text_auto=None,
    hovermode: str = "x unified",
    margin: int | dict | None = None,
    return_fig: bool = False,
    **layout_kwargs,
):
    """
    Bar chart of a 1D tensor, or of several as grouped bars. Extra keyword args go to fig.update_layout.

    Args:
        tensor: 1D values for the bar heights, or several series as grouped bars: either a list of 1D series or a 2D array with one series per row.
        renderer: plotly renderer to show with, e.g. "browser". None uses the default.
        x: labels for the bars, shared by every series. Defaults to 0, 1, 2, ...
        names: legend name per series, in order.
        labels: axis names, e.g. {"x": "Head", "y": "Logit diff"}.
        title: figure title.
        template: plotly theme, e.g. "simple_white".
        size: (height, width) in pixels, shorthand for passing both.
        height, width: figure size in pixels.
        color: values to color the bars by.
        text_auto: write each bar's value on it. True, or a format string like ".2f".
        hovermode: plotly hover behaviour. Defaults to one shared box for all bars at that x.
        margin: padding in pixels. An int sets all four sides.
        return_fig: return the figure instead of showing it.
    """
    arr = to_series(tensor)
    if size is not None:
        height, width = size
    px_kwargs = drop_none(
        x=x, labels=labels, title=title, template=template, height=height, width=width,
        color=color, text_auto=text_auto,
    )
    fig = px.bar(y=arr, **px_kwargs).update_layout(hovermode=hovermode, **layout_kwargs, **margin_dict(margin))
    if names is not None:
        for i in range(len(fig.data)):
            fig.data[i]["name"] = names[i]
    return fig if return_fig else fig.show(renderer=renderer)


def hist(
    tensor,
    renderer=None,
    *,
    names: list[str] | None = None,
    labels: dict | None = None,
    title: str | None = None,
    template: str | None = None,
    size: tuple[int, int] | None = None,
    height: int | None = None,
    width: int | None = None,
    nbins: int | None = None,
    opacity: float | None = None,
    histnorm: str | None = None,
    color=None,
    marginal: str | None = None,
    add_mean_line: bool = False,
    barmode: str = "overlay",
    bargap: float = 0.0,
    hovermode: str = "x unified",
    autosize: bool = False,
    margin: int | dict | None = None,
    static: bool = False,
    return_fig: bool = False,
    **layout_kwargs,
):
    """
    Histogram of a tensor, or of several overlaid. Extra keyword args go to fig.update_layout.

    Args:
        tensor: 1D values to bin, or several series to overlay: either a list of 1D series or a 2D array with one series per row.
        renderer: plotly renderer to show with, e.g. "browser". None uses the default.
        names: legend name per series, in order. With a list of series it is consumed as the traces are named, so pass a list you don't need afterwards.
        labels: axis names, e.g. {"x": "Logit diff", "y": "Count"}.
        title: figure title.
        template: plotly theme, e.g. "simple_white".
        size: (height, width) in pixels, shorthand for passing both.
        height, width: figure size in pixels.
        nbins: number of bins. Plotly picks one by default.
        opacity: bar opacity, 0 to 1. Useful for overlaid histograms.
        histnorm: what the bar heights mean, e.g. "probability" or "percent". Counts by default.
        color: values to split the data into differently-colored histograms by. Only for a single series.
        marginal: extra distribution plot above the bars, e.g. "box" or "rug". Only for a single series.
        add_mean_line: draw a dashed vertical line at each series' mean.
        barmode: how overlapping bars are drawn: "overlay", "group" or "stack".
        bargap: gap between bars, 0 to 1.
        hovermode: plotly hover behaviour. Defaults to one shared box for all series at that x.
        autosize: let the figure resize itself to its container.
        margin: padding in pixels. An int sets all four sides.
        static: show as a non-interactive image.
        return_fig: return the figure instead of showing it.
    """
    arr = to_series(tensor)
    if size is not None:
        height, width = size
    layout = dict(
        barmode=barmode, bargap=bargap, hovermode=hovermode, autosize=autosize,
        modebar_add=['drawline', 'drawopenpath', 'drawclosedpath', 'drawcircle', 'drawrect', 'eraseshape'],
    ) | layout_kwargs | margin_dict(margin)

    # If `arr` has a list of arrays, then just doing px.histogram doesn't work annoyingly enough
    # This is janky, even for my functions!
    if isinstance(arr, list) and isinstance(arr[0], np.ndarray):
        assert marginal is None, "Can't use `marginal` with a list of arrays"
        assert color is None, "Can't use `color` with a list of arrays"
        layout.update(drop_none(title=title, template=template, height=height, width=width))
        if labels is not None:
            layout["xaxis_title_text"] = labels.get("x", "")
            layout["yaxis_title_text"] = labels.get("y", "")
        fig = go.Figure(layout=go.Layout(**layout))
        for x in arr:
            fig.add_trace(go.Histogram(x=x, name=names.pop(0) if names is not None else None, nbinsx=nbins, opacity=opacity, histnorm=histnorm, bingroup="x"))  # bingroup makes the series share bin edges, like px.histogram does
    else:
        px_kwargs = drop_none(
            labels=labels, title=title, template=template, height=height, width=width,
            nbins=nbins, opacity=opacity, histnorm=histnorm, color=color, marginal=marginal,
        )
        fig = px.histogram(x=arr, **px_kwargs).update_layout(**layout)
        if names is not None:
            for i in range(len(fig.data)):
                fig.data[i]["name"] = names[i // 2 if marginal is not None else i]

    if add_mean_line:
        for series in (arr if isinstance(arr, list) else [arr]):
            fig.add_vline(x=series.mean(), line_width=3, line_dash="dash", line_color="black", annotation_text=f"Mean = {series.mean():.3f}", annotation_position="top")
    return fig if return_fig else fig.show(renderer=renderer, config={"staticPlot": static})




# PLOTTING FUNCTIONS FOR PART 2: INTRO TO MECH INTERP

def plot_comp_scores(model, comp_scores, title: str = "", baseline: t.Tensor | None = None) -> go.Figure:
    px.imshow(
        to_numpy(comp_scores),
        y=[f"L0H{h}" for h in range(model.cfg.n_heads)],
        x=[f"L1H{h}" for h in range(model.cfg.n_heads)],
        labels={"x": "Layer 1", "y": "Layer 0"},
        title=title,
        color_continuous_scale="RdBu" if baseline is not None else "Blues",
        color_continuous_midpoint=baseline if baseline is not None else None,
        zmin=None if baseline is not None else 0.0,
    ).show()

def convert_tokens_to_string(model, tokens, batch_index=0):
    '''
    Helper function to convert tokens into a list of strings, for printing.
    '''
    if len(tokens.shape) == 2:
        tokens = tokens[batch_index]
    return [f"|{model.tokenizer.decode(tok)}|_{c}" for (c, tok) in enumerate(tokens)]


def plot_logit_attribution(model, logit_attr: t.Tensor, tokens: t.Tensor, title: str = ""):
    tokens = tokens.squeeze()
    y_labels = convert_tokens_to_string(model, tokens[:-1])
    x_labels = ["Direct"] + [f"L{l}H{h}" for l in range(model.cfg.n_layers) for h in range(model.cfg.n_heads)]
    imshow(
        to_numpy(logit_attr), 
        x=x_labels, y=y_labels, 
        labels={"x": "Term", "y": "Position", "color": "logit"}, title=title if title else None, 
        height=18*len(y_labels), width=24*len(x_labels)
    )







# PLOTTING FUNCTIONS FOR PART 4: INTERP ON ALGORITHMIC MODEL

color_discrete_map = dict(zip(['both failures', 'just neg failure', 'balanced', 'just total elevation failure'], px.colors.qualitative.D3))
# names = ["balanced", "just total elevation failure", "just neg failure", "both failures"]
# colors = ['#2CA02C', '#1c96eb', '#b300ff', '#ff4800']
# color_discrete_map = dict(zip(names, colors))

def plot_failure_types_scatter(
    unbalanced_component_1: Tensor,
    unbalanced_component_2: Tensor,
    failure_types_dict: dict[str, Tensor],
    data
):
    failure_types = np.full(len(unbalanced_component_1), "", dtype=np.dtype("U32"))
    for name, mask in failure_types_dict.items():
        failure_types = np.where(to_numpy(mask), name, failure_types)
    failures_df = pd.DataFrame({
        "Head 2.0 contribution": to_numpy(unbalanced_component_1),
        "Head 2.1 contribution": to_numpy(unbalanced_component_2),
        "Failure type": to_numpy(failure_types),
    })[data.starts_open.tolist()]
    fig = px.scatter(
        failures_df, color_discrete_map=color_discrete_map,
        x="Head 2.0 contribution", y="Head 2.1 contribution", color="Failure type", 
        title="h20 vs h21 for different failure types", template="simple_white", height=600, width=800,
        # category_orders={"color": failure_types_dict.keys()},
    ).update_traces(marker_size=4)
    fig.show()

def plot_contribution_vs_open_proportion(unbalanced_component: Tensor, title: str, failure_types_dict: dict, data):
    failure_types = np.full(len(unbalanced_component), "", dtype=np.dtype("U32"))
    for name, mask in failure_types_dict.items():
        failure_types = np.where(to_numpy(mask), name, failure_types)
    fig = px.scatter(
        x=to_numpy(data.open_proportion), y=to_numpy(unbalanced_component), color=failure_types, color_discrete_map=color_discrete_map,
        title=title, template="simple_white", height=500, width=800,
        labels={"x": "Open-proportion", "y": f"Head {title} contribution"}
    ).update_traces(marker_size=4, opacity=0.5).update_layout(legend_title_text='Failure type')
    fig.show()

def mlp_attribution_scatter(
    out_by_component_in_pre_20_unbalanced_dir: Tensor,
    data, failure_types_dict: dict
) -> None:
    failure_types = np.full(out_by_component_in_pre_20_unbalanced_dir.shape[-1], "", dtype=np.dtype("U32"))
    for name, mask in failure_types_dict.items():
        failure_types = np.where(to_numpy(mask), name, failure_types)
    for layer in range(2):
        mlp_output = out_by_component_in_pre_20_unbalanced_dir[3+layer*3]
        fig = px.scatter(
            x=to_numpy(data.open_proportion[data.starts_open]), 
            y=to_numpy(mlp_output[data.starts_open]), 
            color_discrete_map=color_discrete_map,
            color=to_numpy(failure_types)[to_numpy(data.starts_open)], 
            title=f"Amount MLP {layer} writes in unbalanced direction for Head 2.0", 
            template="simple_white", height=500, width=800,
            labels={"x": "Open-proportion", "y": "Head 2.0 contribution"}
        ).update_traces(marker_size=4, opacity=0.5).update_layout(legend_title_text='Failure type')
        fig.show()

def plot_neurons(neurons_in_unbalanced_dir: Tensor, model, data, failure_types_dict: dict, layer: int, renderer=None):
    
    failure_types = np.full(neurons_in_unbalanced_dir.shape[0], "", dtype=np.dtype("U32"))
    for name, mask in failure_types_dict.items():
        failure_types = np.where(to_numpy(mask[to_numpy(data.starts_open)]), name, failure_types)

    # Get data that can be turned into a dataframe (plotly express is sometimes easier to use with a dataframe)
    # Plot a scatter plot of all the neuron contributions, color-coded according to failure type, with slider to view neurons
    neuron_numbers = einops.repeat(t.arange(model.cfg.d_model), "n -> (s n)", s=data.starts_open.sum())
    failure_types = einops.repeat(failure_types, "s -> (s n)", n=model.cfg.d_model)
    data_open_proportion = einops.repeat(data.open_proportion[data.starts_open], "s -> (s n)", n=model.cfg.d_model)
    df = pd.DataFrame({
        "Output in 2.0 direction": to_numpy(neurons_in_unbalanced_dir.flatten()),
        "Neuron number": to_numpy(neuron_numbers),
        "Open-proportion": to_numpy(data_open_proportion),
        "Failure type": failure_types
    })
    fig = px.scatter(
        df, 
        x="Open-proportion", y="Output in 2.0 direction", color="Failure type", animation_frame="Neuron number",
        title=f"Neuron contributions from layer {layer}", 
        template="simple_white", height=800, width=1100
    ).update_traces(marker_size=3).update_layout(xaxis_range=[0, 1], yaxis_range=[-5, 5])
    fig.show(renderer=renderer)

def plot_attn_pattern(pattern: Tensor):
    fig = px.imshow(
        pattern, 
        title="Estimate for avg attn probabilities when query is from '('",
        labels={"x": "Key tokens (avg of left & right parens)", "y": "Query tokens (all left parens)"},
        height=900, width=900,
        color_continuous_scale="RdBu_r", range_color=[0, pattern.max().item()]
    ).update_layout(
        xaxis = dict(
            tickmode = "array", ticktext = ["[start]", *[f"{i+1}" for i in range(40)], "[end]"],
            tickvals = list(range(42)), tickangle = 0,
        ),
        yaxis = dict(
            tickmode = "array", ticktext = ["[start]", *[f"{i+1}" for i in range(40)], "[end]"],
            tickvals = list(range(42)), 
        ),
    )
    fig.show()
    

def hists_per_comp(out_by_component_in_unbalanced_dir: Tensor, data, xaxis_range=(-1, 1)):
    '''
    Plots the contributions in the unbalanced direction, as supplied by the `out_by_component_in_unbalanced_dir` tensor.
    '''
    titles = {
        (1, 1): "embeddings",
        (2, 1): "head 0.0", (2, 2): "head 0.1", (2, 3): "mlp 0",
        (3, 1): "head `1.0`", (3, 2): "head `1.1`", (3, 3): "mlp 1",
        (4, 1): "head 2.0", (4, 2): "head 2.1", (4, 3): "mlp 2"
    }
    n_layers = out_by_component_in_unbalanced_dir.shape[0] // 3
    fig = make_subplots(rows=n_layers+1, cols=3)
    for ((row, col), title), in_dir in zip(titles.items(), out_by_component_in_unbalanced_dir):
        fig.add_trace(go.Histogram(x=to_numpy(in_dir[data.isbal]), name="Balanced", marker_color="blue", opacity=0.5, legendgroup = '1', showlegend=title=="embeddings"), row=row, col=col)
        fig.add_trace(go.Histogram(x=to_numpy(in_dir[~data.isbal]), name="Unbalanced", marker_color="red", opacity=0.5, legendgroup = '2', showlegend=title=="embeddings"), row=row, col=col)
        fig.update_xaxes(title_text=title, row=row, col=col, range=xaxis_range)
    fig.update_layout(width=1200, height=250*(n_layers+1), barmode="overlay", legend=dict(yanchor="top", y=0.92, xanchor="left", x=0.4), title="Histograms of component significance")
    fig.show()


def plot_loss_difference(log_probs, rep_str, seq_len):
    fig = px.line(
        to_numpy(log_probs), hover_name=rep_str[1:],
        title=f"Per token log prob on correct token, for sequence of length {seq_len}*2 (repeated twice)",
        labels={"index": "Sequence position", "value": "Log prob"}
    ).update_layout(showlegend=False, hovermode="x unified")
    fig.add_vrect(x0=0, x1=seq_len-.5, fillcolor="red", opacity=0.2, line_width=0)
    fig.add_vrect(x0=seq_len-.5, x1=2*seq_len-1, fillcolor="green", opacity=0.2, line_width=0)
    fig.show()
