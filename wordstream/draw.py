import random
import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle, PathPatch
from matplotlib.path import Path

from wordstream.boxes import build_boxes, Box, box_from_row, topic_boxes_to_path
from wordstream.util import WordStreamData, Word, get_max_sudden, load_parlament_data
from wordstream.placement import Placement, WordPlacement


@dataclass
class DrawOptions:
    """Class to set configuration for WordStream Visualisation."""
    width: int
    """Width of visualisation in inches"""
    height: int
    """Height of visualisation in inches"""
    min_font_size: float
    "minimal font size of words placed in visualisation in pt"
    max_font_size: float
    "maximal font size of words placed in visualisation in pt"


def init_word_placement(placement: Placement, word: Word) -> WordPlacement:
    """Initializes WordPlacement by setting the word, the size of the bounding box and the sprite

    Keyword arguments:
    placement: Placement object where to place the Word in.
    word: 'Word' object with text frequency and sudden attention measure.
    """
    wp = WordPlacement(0, 0, 0, 0, 0, word=word)
    placement.get_size(wp)
    placement.get_sprite(wp)
    return wp


def place_topic(placement: Placement, words: pd.Series, topic_boxes: pd.DataFrame, topic_polygon: Path) -> list[dict]:
    """Places words in the boxes of a topic.
    Does not guarantee that all words can be placed

    Keyword Arguments:
    placement -- Figure where to place the words in
    words -- Series of words which should be placed
    topic_boxes -- the boxes where to place the words in
    topic_polygon -- the polygon one get's by stitching the boxes together
    """

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
    """Calculates where the words of WordStreamData are placed and returns result

    Keyword arguments:
    data -- Fully initialized WordStreamData where sudden and frequency is set.
    width -- width in inches of area where words should be placed.
    height -- height in inches of area where words should be placed.
    font_size -- tupel of minimum and maximum font size (in pt) which should be used.
    """

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
    _debug_draw_boxes(ax, boxes, placement, word_placements)
    plt.show(dpi=ppi)

    return word_placements


def placed_in_polygon(topic_polygon: Path, wp: WordPlacement):
    """Checks if WordPlacement is within this polygon"""

    word_box = [(wp.x, wp.y), (wp.x + wp.width, wp.y), (wp.x + wp.width, wp.y + wp.height), (wp.x, wp.y + wp.height)]
    return topic_polygon.contains_path(Path(word_box))


def place(word: WordPlacement, placement: Placement, box: Box, topic_boxes: pd.DataFrame, topic_polygon: Path) -> bool:
    """Tries to place word in figure by searching for unoccupied space in within the given box and polygon.
    Returns True of word could be placed, False if not

    Keyword arguments:

    word -- Word which should be placed, Coordinates and placed attribute are set if placed sucessfully
    placement -- Image where word should be placed in. Is used to check if word collides with any other words
    box -- Box where word should be placed
    topic_polygon -- Path for a topic, used to see if word can be placed fully in area of the given topic
    """
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
    """Function to calculate an Archemedean Spiral
    initialized with a maximum size.

    returns function where next position can be calculated"""
    e = size[0] / size[1]

    def spiral(t):
        return [e * (t * 0.1) * math.cos(t), t * math.sin(t)]

    return spiral


def rectangularSpiral(size):
    """Function to calculate a rectangular Spiral
    initialized with a maximum size.

    returns function where next position can be calculated
    """
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


def _debug_draw_boxes(ax, boxes: dict[str, pd.DataFrame], placement: Placement, placements: dict[str, list]):
    """Draws boxes - used for debugging to see if boxes are drawn properly"""
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



def draw_parlament(options: DrawOptions, legislative_periods: list[str], fulltext: bool = False) -> tuple[dict, WordStreamData]:
    """ Loads the data for given legislative periods, places them in figure generated by Drawoptions
    returns the placements and WordStreamData

    Keyword arguments:
    options -- Configuration for size of figure and font
    legislative_periods -- List of roman numerals to indicate what data should be loaded
    fulltext -- True: fulltext of motions is drawn; otherwise eurovoc keywords are used.
    """
    data = load_parlament_data(legislative_periods, fulltext=fulltext)
    placement = place_words(data, options.width, options.height, font_size=(options.min_font_size, options.max_font_size))
    return placement, data


if __name__ == '__main__':
    options = DrawOptions(width=24, height=12, min_font_size=15, max_font_size=35)
    draw_parlament(options)

