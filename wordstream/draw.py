from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from wordstream.boxes import build_boxes
from wordstream.util import WordStreamData, load_fact_check, Word


@dataclass
class WordPlacement:
    x: float
    y: float
    width: float
    height: float
    word: Word


def place_words(data: WordStreamData, width: int, sizing_func: Callable[[WordPlacement], WordPlacement]):
    boxes = build_boxes(data, width)


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

if __name__ == '__main__':
    data = load_fact_check()
    boxes = build_boxes(data, 1000)
    test_placement = WordPlacement(x=10, y=-10, width=30, height=5, word=Word(text="", frequency=0, sudden=0))
    p = placed_in_box(boxes, "person", test_placement)
    print(p)

