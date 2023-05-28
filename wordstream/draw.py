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
from wordstream.util import WordStreamData, load_fact_check
from wordstream.placement import Placement, WordPlacement


def place_words(data: WordStreamData, width: int, height: int, sizing_func: Callable[[WordPlacement], WordPlacement]):
    ppi = 100
    boxes = build_boxes(data, width, height)
    placement = Placement(round(width/ppi), round(height/ppi), ppi, 5, 10,30,"../fonts/RobotoMono-VariableFont_wght.ttf")
    test_topic = data.topics[0]
    first_box = boxes[test_topic].iloc[0]
    box_to_place_in = box_from_row(first_box)
    words_for_box = data.df.loc[0,test_topic]
    for word in words_for_box:
        w_placement = WordPlacement(0,0,0,0,0,word=word)
        w_placement = placement.get_size(w_placement)
        w_placement = placement.get_sprite(w_placement)
        w_placement.x = placement.inv_box_width_map(w_placement.x)
        w_placement.y = placement.inv_box_height_map(w_placement.y)
        w_placement = place(w_placement, placement, box_to_place_in)
        placement.place(w_placement)
        print(f'set {(np.asarray(placement.img)>0).sum()} pixeles of the array')

    print("")



def placed_in_box(boxes: dict[str, pd.DataFrame], topic: str, word: WordPlacement, check_individual: bool = False):
    topic_box = boxes[topic]
    x = topic_box.index.values
    box_idx = np.where(x <= word.x)[0].max(initial=0)
    box_x = x[box_idx]
    box = topic_box.loc[box_x]

    # if first box check if word is outside of box
    if box_x == x[0] and word.x < box_x:
        return False

    # if last box of we only consider an individual box, check if word end is outside of box
    if (box_x == x[-1] or check_individual) and word.x + word.width > box_x + box.width:
        return False

    word_bottom_in_box = box.y <= word.y <= box.y + box.height
    word_top_in_box = box.y <= word.y + word.height <= box.y + box.height

    if word_bottom_in_box and word_top_in_box:
        return True
    else:
        return False


def place(word: WordPlacement, placement: Placement, box: Box):
    bw = placement.width
    bh = placement.height
    maxDelta = int((box.width * box.width + box.height * box.height) ** 0.5)
    startX = int(box.x + (box.width * (random.random() + .5) /2))
    startY = int(box.y + (box.height * (random.random() + .5) /2))
    s = achemedeanSpiral([box.width, box.height])
    dt = 1 if random.random() < .5 else -1
    t = -dt
    dxdy, dx, dy = None, None, None
    word.x = startX
    word.y = startY
    word.placed = False

    while True:
        t+=dt
        dxdy = s(t)
        if not dxdy:
            break

        dx = int(dxdy[0])
        dy = int(dxdy[1])

        if max(abs(dx), abs(dy)) >= maxDelta:
            break

        word.x = startX + dx
        word.y = startY + dy

        print(f'Try to place word {word.word.text} at x: {word.x}, y: {word.y}')

        if word.x < 0 or word.y < -(placement.height/2) or word.x + word.width > placement.width or word.y + word.height > placement.height/2:
            continue
        if placement.check_placement(word):
            break
    return word


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


def debug_draw_boxes(ax, boxes: dict[str, pd.DataFrame]):
    for tb, col in zip(boxes.items(), ["red", "green", "blue", "purple"]):
        topic, topic_boxes = tb
        for x in topic_boxes.index:
            box = topic_boxes.loc[x]
            ax.add_patch(Rectangle((x, box.y), box.width, box.height, color=col))


if __name__ == '__main__':
    data = load_fact_check()

    place_words(data, 1000, 400, None)




    # boxes = build_boxes(data, 1000, 400)
    # fig, ax = plt.subplots(1, 1)
    # debug_draw_boxes(ax, boxes)
    # ax.set_xlim(0, 1000)
    # ax.set_ylim(-200, 200)
    # plt.show()
    # test_placement = WordPlacement(x=10, y=-10, width=30, height=5, word=Word(text="", frequency=0, sudden=0))
    # p = placed_in_box(boxes, "person", test_placement)
    # print(p)
