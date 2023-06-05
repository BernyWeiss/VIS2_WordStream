from dataclasses import dataclass
from typing import Callable

import random
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from PIL import Image, ImageDraw, ImageFont

from wordstream.boxes import build_boxes, Box, box_from_row
from wordstream.util import WordStreamData, load_fact_check, Word, get_max_sudden
from wordstream.placement import Placement, WordPlacement


def init_word_placement(placement: Placement, word: Word) -> WordPlacement:
    wp = WordPlacement(0, 0, 0, 0, 0, word=word)
    placement.get_size(wp)
    placement.get_sprite(wp)
    return wp


def place_topic(placement: Placement, words: pd.Series, topic_boxes: pd.DataFrame):
    # we place most frequent words of each box first, then second etc. -> create iterator for each list of words (map)
    word_placements = words.apply(lambda ws: map(lambda w: init_word_placement(placement, w), ws)).tolist()
    n_words = words.apply(lambda ws: len(ws)).sum()

    words_tried = 0
    while words_tried < n_words:
        # perform run over next most frequent words in each box
        for i, words_in_box in enumerate(word_placements):
            try:
                word_placement = next(words_in_box)
                words_tried += 1
            except StopIteration:
                continue
            place(word_placement, placement, box=box_from_row(topic_boxes.iloc[i]), topic_boxes=topic_boxes)


def place_words(data: WordStreamData, width: int, height: int, font_size=tuple[float, float]):
    min_font, max_font = font_size
    ppi = 200
    boxes = build_boxes(data, width, height)
    max_sudden = get_max_sudden(data)
    placement = Placement(width, height, ppi, max_sudden, min_font, max_font, "../fonts/RobotoMono-VariableFont_wght.ttf")
    for topic in data.topics:
        place_topic(placement, data.df[topic], boxes[topic])

    fig, ax = plt.subplots(1, 1, figsize=(width, height))
    ax.imshow(np.asarray(placement.img))
    debug_draw_boxes(ax, boxes, placement)
    plt.show(dpi=ppi)

    return placement


def check_word_in_box(word: WordPlacement, box_idx: int, x: np.ndarray, boxes: pd.DataFrame, check_individual: bool):
    box_x = x[box_idx]
    box = boxes.loc[box_x]

    # if first box check if word is outside of box
    if box_x == x[0] and word.x < box_x:
        return False

    # if last box, or we only consider an individual box, check if word end is outside of single box
    if (box_x == x[-1] or check_individual) and word.x + word.width > box_x + box.width:
        return False

    word_bottom_in_box = box.y <= word.y <= box.y + box.height
    word_top_in_box = box.y <= word.y + word.height <= box.y + box.height

    return word_bottom_in_box and word_top_in_box


def placed_in_box(topic_boxes: pd.DataFrame, word: WordPlacement, check_individual: bool = False):
    x = topic_boxes.index.values
    box_idx = np.where(x <= word.x)[0].max(initial=0)
    box_nxt = np.where(x <= word.x + word.width)[0].max(initial=0)

    word_start_in_box = check_word_in_box(word, box_idx, x, topic_boxes, check_individual)
    word_end_in_box = check_word_in_box(word, box_nxt, x, topic_boxes, check_individual) if box_idx != box_nxt else True
    return word_start_in_box and word_end_in_box


def place(word: WordPlacement, placement: Placement, box: Box, topic_boxes: pd.DataFrame):
    maxDelta = (box.width * box.width + box.height * box.height) ** 0.5
    startX = box.x + (box.width * (random.random() + .5) / 2)
    startY = box.y + (box.height * (random.random() + .5) / 2)
    s = achemedeanSpiral([box.width, box.height])
    dt = 1 if random.random() < .5 else -1
    dt *= word.height
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

        # print(f'Try to place word {word.word.text} at x: {word.x}, y: {word.y}')

        # check if word is placed inside the canvas first
        if word.x < 0 or word.y < -placement.height / 2 or word.x + word.width > placement.width or word.y + word.height > placement.height / 2:
            continue
        # also check if word is placed inside the current box first
        if not placed_in_box(topic_boxes, word, check_individual=False):
            continue

        if placement.check_placement(word):
            placement.place(word)
            print(f"Success placing {word.word.text} with {(word.sprite > 0).sum()} pixels ")
            break


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


def debug_draw_boxes(ax, boxes: dict[str, pd.DataFrame], placement: Placement):
    for tb, col in zip(boxes.items(), ["red", "green", "blue", "purple"]):
        topic, topic_boxes = tb
        for x in topic_boxes.index:
            box = box_from_row(topic_boxes.loc[x])
            x_px = placement.width_map(box.x)
            y_px = placement.height_map(box.y)
            height_px = placement.box_height_map(box.height)
            width_px = placement.box_width_map(box.width)
            ax.add_patch(Rectangle((x_px, y_px), width_px, height_px, edgecolor=col, facecolor="none", lw=2))


if __name__ == '__main__':
    data = load_fact_check()
    img = place_words(data, 30, 12, font_size=(5., 40.))

