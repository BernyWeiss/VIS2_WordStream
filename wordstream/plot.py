import random
from pathlib import Path

import numpy as np
import pandas as pd
from bokeh.models import LegendItem, GlyphRenderer, Axis, DatetimeTickFormatter, DatetimeTicker, LinearAxis, Grid, \
    MultiLine

from wordstream.boxes import build_boxes
from wordstream.draw import DrawOptions, draw_parlament

font_path = Path("../fonts/RobotoMono-VariableFont_wght.ttf")
"""Path where font for WordStreamVis is stored """

s = 43
random.seed(s)
np.random.seed(s)

PARTY_COLOR = {
    "F": "blue",
    "G": "limegreen",
    "L": "gold",
    "S": "red",
    "V": "black",
    "B": "darkorange",
    "A": "purple",
    "T": "orange",
    "N": "magenta",
    "PJ": "grey"
}
"""Color Mapping of fractions"""

PARTY_NAME = {
    "F": "FPÖ",
    "G": "Grüne",
    "L": "Liberales Forum",
    "S": "SPÖ",
    "V": "ÖVP",
    "B": "BZÖ",
    "A": "ohne Klub",
    "T": "Team Stronach",
    "N": "NEOS",
    "PJ": "PILZ/JETZT"
}
"""Full name of fractions"""

GOVERNMENTS = {
    "XX": "SPÖ/ÖVP",
    "XXI": "ÖVP/FPÖ",
    "XXII": "ÖVP/FPÖ",
    "XXIII": "ÖVP/FPÖ + SPÖ/ÖVP",
    "XXIV": "SPÖ/ÖVP",
    "XXV": "SPÖ/ÖVP",
    "XXVI": "SPÖ/ÖVP + ÖVP/FPÖ",
    "XXVII": "ÖVP/Grüne",
}
"""Governments during legislative periods of parliament"""


def pt_to_px(ppi, pt: float) -> float:
    """Converts pt to pixel"""
    # 1 pt = 1/72 inches
    inches = pt / 72
    return inches * ppi


def px_to_pt(ppi, px: int) -> float:
    """Convert pixel to pt"""
    inches = px / ppi
    return inches * 72


assert pt_to_px(100, px_to_pt(100, 10)) == 10


def plot_bokeh(options: DrawOptions, legislative_periods: list[str], fulltext: bool = False) -> tuple[str, str]:
    """Plots WordStream visualisation in bokeh and generates standalone html document

    Keyword arguments:
    options -- DrawOptions for bokeh plot
    legislative_periods -- roman numerals of periods which should be plotted
    fulltext -- true: fulltext of motions is used, false: eurovoc keywords are used
    """

    from bokeh.io import curdoc, show
    from bokeh.models import ColumnDataSource, Plot, Text, Range1d, HoverTool, Circle, Legend, FixedTicker
    from bokeh.core.properties import value

    placements, data = draw_parlament(options, legislative_periods=legislative_periods, fulltext=fulltext)
    topics = list(placements.keys())

    ppi = 72
    plot = Plot(
        title=None,
        # width=options.width*ppi, height=options.height*ppi,
        frame_width=options.width * ppi, frame_height=options.height * ppi,
        # min_width=options.width * ppi, min_height=options.height * ppi,
        min_border=0, toolbar_location=None,
        background_fill_color=None, border_fill_color=None, background_fill_alpha=0.0
    )
    plot.output_backend = "svg"

    hover = HoverTool()
    hover.tooltips = """
        <div>
            <div>
                <span style="font-size: 17px; font-weight: bold;">@text</span><br>
                <span style="font-size: 15px; font-weight: bold; color: @col">@topic</span>
            </div>
        </div>
    """
    plot.add_tools(hover)

    # creade boxes and add x-axis
    boxes = build_boxes(data, options.width, options.height)
    x_index = boxes[list(boxes.keys())[0]].index.values.tolist()
    label_dict = {}
    for i, s in zip(x_index, data.df["Datum"]):
        label_dict[i] = str(s.year)

    xaxis = LinearAxis(ticker=FixedTicker(ticks=x_index))
    plot.add_layout(xaxis, 'below')
    plot.xaxis[0].major_label_overrides = label_dict
    plot.add_layout(Grid(dimension=0, ticker=xaxis.ticker))

    # Find x-axis positions of start of legislative period
    # xpos = [list(filter(lambda x: label_dict[x] == str(timestamp.year), label_dict))[0] for period, timestamp in
    #        data.segment_start.items()]
    # generate datasource for lines for legislative periods

    x_pos_list = list(label_dict.keys())
    dates_list = list(label_dict.values())

    period_xs = []
    period_ys = []
    period_label = []
    for period, timestamp in data.segment_start.items():
        startidx = dates_list.index(str(timestamp.year))
        if startidx == 0:
            x_pos = x_pos_list[startidx]
        elif len(dates_list) == startidx + 1:
            # legislative period started in current year. (use the with of last year to calculate position
            year_width = options.width - x_pos_list[startidx]
            day_of_year = int(timestamp.strftime("%j"))
            x_pos = x_pos_list[startidx] + year_width * day_of_year / 365
        else:
            year_width = x_pos_list[startidx + 1] - x_pos_list[startidx]
            day_of_year = int(timestamp.strftime("%j"))
            x_pos = x_pos_list[startidx] + year_width * day_of_year / 365
        period_xs.append([x_pos, x_pos])
        period_ys.append([-options.height / 2, options.height / 2])
        period_label.append(period)

    segments = ColumnDataSource(dict(
        xs=period_xs,
        ys=period_ys,
        text=period_label,
        topic=[f"Gesetzgebungsperiode {p} ({GOVERNMENTS[p]})" for p in period_label]
    ))
    segments_text = ColumnDataSource(dict(
        x=[xs[0] for xs in period_xs],
        y=[ys[1] for ys in period_ys],
        text=period_label,
        topic=[f"Gesetzgebungsperiode {p} ({GOVERNMENTS[p]})" for p in period_label]
    ))

    # segments = ColumnDataSource(dict(
    #    xs=[[x, x] for x in xpos],
    #    ys=[[-options.height / 2, options.height / 2] for period in data.segment_start.keys()]))

    period_segments = MultiLine(xs="xs", ys="ys", line_color="black", line_width=1, line_dash="dashed")
    plot.add_glyph(segments, period_segments)
    period_text = Text(x='x', y='y', text='text', text_font_style='bold')
    plot.add_glyph(segments_text, period_text)

    topic_glyphs = dict()
    for topic in topics:
        col = PARTY_COLOR[topic]
        df = pd.DataFrame.from_records(placements[topic])
        if df.size == 0:
            # No topics of club could be placed
            continue
        df["font_size"] = df["font_size"].apply(lambda v: f"{pt_to_px(ppi, v)}px")
        df["topic"] = PARTY_NAME[topic]
        df["col"] = col

        glyph = Text(x="x", y="y", text="text", text_color=col, text_font_size="fs", text_baseline="top",
                     text_align="left")
        glyph.text_font = value("Rubik")
        ds = ColumnDataSource(
            dict(x=df.x, y=df.y, text=df.text, fs=df.font_size, topic=df.topic, col=df.col))
        plot.add_glyph(ds, glyph)

        # circles need to actually be drawn so place them outside of plot area as a hack
        ds_legend = ColumnDataSource(dict(col=df.col, x=df.x, y=df.y + 100))
        points = Circle(x="x", y="y", fill_color="col")
        legend_renderer = plot.add_glyph(ds_legend, points)
        topic_glyphs[PARTY_NAME[topic]] = legend_renderer

    legend = Legend(
        items=[(p, [t]) for p, t in topic_glyphs.items()],
        location="center", orientation="vertical",
        border_line_color="black",
        title='Party',
        background_fill_alpha=0.0
    )
    plot.add_layout(legend, "left")

    plot.y_range = Range1d(options.height / 2, -options.height / 2)
    plot.x_range = Range1d(0, max(x_index) + x_index[1])

    curdoc().add_root(plot)

    from bokeh.embed import components
    plot_script, div = components(plot, wrap_script=True)
    return plot_script, div


if __name__ == '__main__':
    options = DrawOptions(width=11, height=6, min_font_size=10, max_font_size=25)
    # legislative_periods = ["XX", "XXI", "XXII","XXIII", "XXIV","XXV","XXVI", "XXVII"]
    plots = {
        "plot_1": plot_bokeh(options, ["XX", "XXI", "XXII"], fulltext=False),
        "plot_2": plot_bokeh(options, ["XX", "XXI", "XXII"], fulltext=True),
        "plot_3": plot_bokeh(options, ["XXIII", "XXIV", "XXV"], fulltext=False),
        "plot_4": plot_bokeh(options, ["XXVI", "XXVII"], fulltext=False),
        "plot_5": plot_bokeh(DrawOptions(width=32, height=10, min_font_size=10, max_font_size=25),
                             ["XX", "XXI", "XXII", "XXIII", "XXIV", "XXV", "XXVI", "XXVII"],
                             fulltext=False),
    }

    import jinja2

    with open("../fonts/template_custom.html", "r") as f:
        template = jinja2.Template(f.read())

    html = template.render(**plots)
    with open("../html/index.html", "w") as f:
        f.write(html)
