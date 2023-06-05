import random
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle, PathPatch
from matplotlib.path import Path

from wordstream.boxes import build_boxes, Box, box_from_row, topic_boxes_to_path
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
    words_placed = 0
    while words_tried < n_words:
        # perform run over next most frequent words in each box
        for i, words_in_box in enumerate(word_placements):
            try:
                word_placement = next(words_in_box)
                words_tried += 1
            except StopIteration:
                continue
            placed = place(word_placement, placement, box=box_from_row(topic_boxes.iloc[i]), topic_boxes=topic_boxes)
            words_placed += placed
    print(f"Placed {words_placed}/{n_words} in topic!")


def place_topic_greedy(placement: Placement, words: pd.Series, topic_boxes: pd.DataFrame, topic_polygon: Path):
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

def place_words(data: WordStreamData, width: int, height: int, font_size=tuple[float, float]):
    min_font, max_font = font_size
    ppi = 200
    boxes = build_boxes(data, width, height)
    max_sudden = get_max_sudden(data)
    placement = Placement(width, height, ppi, max_sudden, min_font, max_font, "../fonts/RobotoMono-VariableFont_wght.ttf")
    for topic in data.topics:
        topic_polygon = topic_boxes_to_path(boxes[topic])
        place_topic_greedy(placement, data.df[topic], boxes[topic], topic_polygon)

    fig, ax = plt.subplots(1, 1, figsize=(width, height))
    ax.imshow(np.asarray(placement.img))
    debug_draw_boxes(ax, boxes, placement)
    plt.show(dpi=ppi)

    return placement


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


def debug_draw_boxes(ax, boxes: dict[str, pd.DataFrame], placement: Placement):
    for tb, col in zip(boxes.items(), ["red", "green", "blue", "purple"]):
        topic, topic_boxes = tb
        # box_path = topic_boxes_to_path(topic_boxes)
        # max_x = box_path.get_extents().max[0]
        # from matplotlib.transforms import BboxTransform
        # box_transformed = box_path.transformed(BboxTransform(boxin=box_path.get_extents(), boxout=ax.dataLim))
        # ax.add_patch(PathPatch(box_transformed, edgecolor=col, facecolor="none", lw=2))
        for x in topic_boxes.index:
            box = box_from_row(topic_boxes.loc[x])
            x_px = placement.width_map(box.x)
            y_px = placement.height_map(box.y)
            height_px = placement.box_height_map(box.height)
            width_px = placement.box_width_map(box.width)
            ax.add_patch(Rectangle((x_px, y_px), width_px, height_px, edgecolor=col, facecolor="none", lw=2))


if __name__ == '__main__':
    data = load_fact_check()
    img = place_words(data, 24, 12, font_size=(15., 35.))

