from pathlib import Path

from wordstream.draw import draw_fact_check, DrawOptions

font_path = Path("../fonts/RobotoMono-VariableFont_wght.ttf")

def plot_matplotlib():
    import matplotlib.pyplot as plt

    options = DrawOptions(width=24, height=12, min_font_size=15, max_font_size=45)
    placements = draw_fact_check(options)
    topics = list(placements.keys())

    ppi = 100
    fig, ax = plt.subplots(1, 1, figsize=(options.width, options.height))
    for topic, col in zip(topics, ["red", "green", "blue", "purple"]):
        for word in placements[topic]:
            # apparently matplotlib also uses pixels? use ppi of plot to calculate pixels
            font_size = 72*word["font_size"]/ppi
            ax.text(x=word["x"], y=word["y"], s=word["text"], color=col, font=font_path, fontsize=font_size, ha="left", va="top")

    ax.set_xlim(0, options.width)
    ax.set_ylim(options.height/2, -options.height/2)
    fig.set_dpi(ppi)

    fig.show()


def plot_bokeh():
    from bokeh.io import curdoc, show
    from bokeh.models import ColumnDataSource, Grid, LinearAxis, Plot, Text, Range1d

    options = DrawOptions(width=24, height=12, min_font_size=15, max_font_size=45)
    placements = draw_fact_check(options)
    topics = list(placements.keys())

    ppi = 72  # so that pixel and pt are equal
    plot = Plot(
        title=None, width=options.width*ppi, height=options.height*ppi,
        min_border=0, toolbar_location=None)

    for topic, col in zip(topics, ["red", "green", "blue", "purple"]):
        for word in placements[topic]:
            # TODO: change font (or generate plot with helvetica)
            glyph = Text(x="x", y="y", text="text", text_color=col, text_font_size="fs", text_font="f", text_baseline="top", text_align="left")
            ds = ColumnDataSource(dict(x=[word["x"]], y=[word["y"]], text=[word["text"]], fs=[f'{word["font_size"]}px'], f=["helvetica"]))
            plot.add_glyph(ds, glyph)

    plot.y_range = Range1d(options.height/2, -options.height/2)

    curdoc().add_root(plot)
    show(plot)


if __name__ == '__main__':
    plot_bokeh()

