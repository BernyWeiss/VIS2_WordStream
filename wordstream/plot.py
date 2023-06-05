import matplotlib.pyplot as plt
from pathlib import Path

from wordstream.draw import draw_fact_check, DrawOptions

font_path = Path("../fonts/RobotoMono-VariableFont_wght.ttf")

def plot_matplotlib():
    options = DrawOptions(width=24, height=12, min_font_size=15, max_font_size=35)
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


if __name__ == '__main__':
    plot_matplotlib()

