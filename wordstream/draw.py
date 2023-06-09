import random
import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle, PathPatch
from matplotlib.path import Path

from wordstream.boxes import build_boxes, Box, box_from_row, topic_boxes_to_path
from wordstream.util import WordStreamData, load_fact_check, Word, get_max_sudden, load_parlament_data
from wordstream.placement import Placement, WordPlacement


@dataclass
class DrawOptions:
    width: int
    height: int
    min_font_size: float
    max_font_size: float


def init_word_placement(placement: Placement, word: Word) -> WordPlacement:
    wp = WordPlacement(0, 0, 0, 0, 0, word=word)
    placement.get_size(wp)
    placement.get_sprite(wp)
    return wp


def place_topic(placement: Placement, words: pd.Series, topic_boxes: pd.DataFrame, topic_polygon: Path) -> list[dict]:
    # place all words in the first box then second and so on
    word_placements = words.apply(lambda ws: list(map(lambda w: init_word_placement(placement, w), ws))).tolist()
    n_words = words.apply(lambda ws: len(ws)).sum()

    words_tried = 0
    words_placed = 0
    while words_tried < n_words:
        # perform run over next most frequent words in each box
        for i, words_in_box in enumerate(word_placements):
            for word_placement in words_in_box:
                words_tried += 1
                placed = place(word_placement, placement, box=box_from_row(topic_boxes.iloc[i]), topic_boxes=topic_boxes, topic_polygon=topic_polygon)
                words_placed += placed

    print(f"Placed {words_placed}/{n_words} in topic!")
    placements_flat = [w for ws in word_placements for w in ws]
    return list(map(lambda w: w.to_dict(), filter(lambda w: w.placed, placements_flat)))


def place_words(data: WordStreamData, width: int, height: int, font_size=tuple[float, float]) -> dict:
    min_font, max_font = font_size
    ppi = 200
    boxes = build_boxes(data, width, height)
    max_sudden = get_max_sudden(data)
    placement = Placement(width, height, ppi, max_sudden, min_font, max_font, "../fonts/Rubik-Medium.ttf")
    word_placements = dict()
    for topic in data.topics:
        topic_polygon = topic_boxes_to_path(boxes[topic])
        word_placements[topic] = place_topic(placement, data.df[topic], boxes[topic], topic_polygon)

    fig, ax = plt.subplots(1, 1, figsize=(width, height))
    ax.imshow(np.asarray(placement.img))
    debug_draw_boxes(ax, boxes, placement, word_placements)
    plt.show(dpi=ppi)

    return word_placements


def placed_in_polygon(topic_polygon: Path, wp: WordPlacement):
    word_box = [(wp.x, wp.y), (wp.x + wp.width, wp.y), (wp.x + wp.width, wp.y + wp.height), (wp.x, wp.y + wp.height)]
    return topic_polygon.contains_path(Path(word_box))


def place(word: WordPlacement, placement: Placement, box: Box, topic_boxes: pd.DataFrame, topic_polygon: Path) -> bool:
    maxDelta = (box.width * box.width + box.height * box.height) ** 0.5
    startX = box.x + (box.width * (random.random() + .5) / 2)
    startY = box.y + (box.height * (random.random() + .5) / 2)
    s = achemedeanSpiral([box.width, box.height])
    dt = 1 if random.random() < .5 else -1
    dt *= 0.5 * word.height
    t = -dt
    dxdy, dx, dy = None, None, None
    word.x = startX
    word.y = startY
    word.placed = False

    while True:
        t += dt
        dxdy = s(t)
        if not dxdy:
            break

        dx = dxdy[0]
        dy = dxdy[1]

        if max(abs(dx), abs(dy)) >= maxDelta:
            break

        word.x = startX + dx
        word.y = startY + dy

        # check if word is placed inside the canvas first
        if word.x < 0 or word.y < -placement.height / 2 or word.x + word.width > placement.width or word.y + word.height > placement.height / 2:
            continue
        # also check if word is placed inside the current box first
        if not placed_in_polygon(topic_polygon, word):
            continue

        if placement.check_placement(word):
            placement.place(word)
            # print(f"Success placing {word.word.text} with {(word.sprite > 0).sum()} pixels ")
            return True

    return False


def achemedeanSpiral(size):
    e = size[0] / size[1]

    def spiral(t):
        return [e * (t * 0.1) * math.cos(t), t * math.sin(t)]

    return spiral


def rectangularSpiral(size):
    dy = 4
    dx = dy * size[0] / size[1]
    x = 0
    y = 0

    def spiral(t):
        sign = -1 if t < 0 else 1
        switch = (int(math.sqrt(1 + 4 * sign * t)) - sign) & 3
        nonlocal x, y
        if switch == 0:
            x += dx
        elif switch == 1:
            y += dy
        elif switch == 2:
            x -= dx
        else:
            y -= dy

    return spiral


spirals = {
    'achemedean': achemedeanSpiral,
    'rectangular': rectangularSpiral,
}


def debug_draw_boxes(ax, boxes: dict[str, pd.DataFrame], placement: Placement, placements: dict[str, list]):
    for tb, col in zip(boxes.items(), ["red", "green", "blue", "purple"]):
        topic, topic_boxes = tb
        for x in topic_boxes.index:
            box = box_from_row(topic_boxes.loc[x])
            x_px = placement.width_map(box.x)
            y_px = placement.height_map(box.y)
            height_px = placement.box_height_map(box.height)
            width_px = placement.box_width_map(box.width)
            ax.add_patch(Rectangle((x_px, y_px), width_px, height_px, edgecolor=col, facecolor="none", lw=2))
    for topic, words in placements.items():
        for word in words:
            x_px = placement.width_map(word["x"])
            y_px = placement.height_map(word["y"])
            ax.plot(x_px,y_px, 'ro')


def draw_fact_check(options: DrawOptions) -> dict:
    data = load_fact_check()
    placement = place_words(data, options.width, options.height, font_size=(options.min_font_size, options.max_font_size))
    return placement

def draw_parlament(options: DrawOptions) -> tuple[dict, WordStreamData]:
    data = load_parlament_data(["XX", "XXI"])
    placement = place_words(data, options.width, options.height, font_size=(options.min_font_size, options.max_font_size))
    return placement, data


if __name__ == '__main__':
    options = DrawOptions(width=24, height=12, min_font_size=15, max_font_size=35)
    draw_parlament(options)
    #draw_fact_check(options)

