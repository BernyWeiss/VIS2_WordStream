import random
from pathlib import Path

import numpy as np
import pandas as pd
from bokeh.embed import file_html
from bokeh.io import save
from bokeh.resources import CDN

from wordstream.draw import draw_fact_check, DrawOptions

font_path = Path("../fonts/RobotoMono-VariableFont_wght.ttf")

s = 43
random.seed(s)
np.random.seed(s)

def pt_to_px(ppi, pt: float) -> float:
    # 1 pt = 1/72 inches
    inches = pt / 72
    return inches * ppi


def px_to_pt(ppi, px: int) -> float:
    inches = px / ppi
    return inches * 72


assert pt_to_px(100, px_to_pt(100, 10)) == 10

def font_size_conversion(ppi, font_size, ppi_base=72):
    """ I think the libraries use  """
    px = pt_to_px(ppi, font_size)
    pt = px_to_pt(ppi_base, int(px))
    return pt


def plot_matplotlib():
    import matplotlib.pyplot as plt
    import matplotlib

    options = DrawOptions(width=24, height=12, min_font_size=15, max_font_size=45)
    placements = draw_fact_check(options)
    topics = list(placements.keys())

    # matplotlib.use("svg", force=True)

    ppi = 72
    fig, ax = plt.subplots(1, 1, figsize=(options.width, options.height))
    for topic, col in zip(topics, ["red", "green", "blue", "purple"]):
        for word in placements[topic]:
            # apparently matplotlib also uses pixels? use ppi of plot to calculate pixels
            font_size = pt_to_px(ppi, word["font_size"])
            # font_size = word["font_size"]
            ax.text(x=word["x"], y=word["y"], s=word["text"], color=col, font=font_path, fontsize=font_size, ha="left", va="top")
            ax.plot(word["x"], word["y"], 'ro')

    ax.set_xlim(0, options.width)
    ax.set_ylim(options.height/2, -options.height/2)

    plt.show(dpi=ppi)
    # plt.savefig("../../mat_img.svg")


def plot_bokeh():
    from bokeh.io import curdoc, show
    from bokeh.models import ColumnDataSource, Plot, Text, Range1d, HoverTool

    options = DrawOptions(width=24, height=12, min_font_size=15, max_font_size=45)
    placements = draw_fact_check(options)
    topics = list(placements.keys())

    ppi = 72
    plot = Plot(
        title=None, width=options.width*ppi, height=options.height*ppi,
        min_border=0, toolbar_location=None)

    hover = HoverTool()
    hover.tooltips = """
        <div>
            <div>
                <span style="font-size: 17px; font-weight: bold;">@text</span><br>
                <span style="font-size: 15px; font-weight: bold; color: @col">@topic</span>
            </div>
            <div>
                <span style="font-size: 15px;">Location</span>
                <span style="font-size: 10px; color: #696;">($x, $y)</span>
            </div>
        </div>
    """
    plot.add_tools(hover)

    for topic, col in zip(topics, ["red", "green", "blue", "purple"]):
        from bokeh.core.properties import value, field
        for word in placements[topic]:
            font_size = pt_to_px(ppi, word["font_size"])
            glyph = Text(x="x", y="y", text="text", text_color=col, text_font_size="fs", text_baseline="top", text_align="left")
            glyph.text_font = value("Roboto Mono")
            ds = ColumnDataSource(dict(x=[word["x"]], y=[word["y"]], text=[word["text"]], fs=[f'{font_size}px'], topic=[topic], col=[col]))
            plot.add_glyph(ds, glyph)

    plot.y_range = Range1d(options.height/2, -options.height/2)

    curdoc().add_root(plot)
    from bokeh.core.templates import FILE
    import jinja2
    with open("../fonts/template.html", "r") as f:
        template = jinja2.Template(f.read())
    html = file_html(plot, CDN, "plot", template=template)
    with open("plot.html", "w") as f:
        f.write(html)


if __name__ == '__main__':
    plot_bokeh()

