import pandas as pd
from dataclasses import dataclass

from wordstream.util import WordStreamData, load_fact_check


@dataclass
class Box:
    x: float
    y: float
    width: float
    height: float


def boxes(data: WordStreamData):
    pass


def build_boxes(data: WordStreamData, width: int) -> dict[str, pd.DataFrame]:
    tf = total_frequencies(data)
    num_boxes = len(data.df)
    box_width = width // num_boxes

    # only create index containing x-values -> y values are just frequencies
    x_index = tf.index.map(lambda i: (i * box_width) + (box_width >> 1))
    tf = tf.set_index(x_index)

    layers = tf.cumsum(axis=1)
    # center frequencies to create silhouette i.e. from (0,max) -> (-max/2,max/2) for each row (max is last column)
    layers -= layers.iloc[:, -1] / 2

    # todo: create dict of topics and represent boxes as dataframe with [y, width, height] as columns and x is index


def total_frequencies(data: WordStreamData) -> pd.DataFrame:
    return data.df[data.topics].apply(lambda topic: topic.apply(lambda words: sum([w.frequency for w in words])))


if __name__ == '__main__':
    data = load_fact_check()
    build_boxes(data, 1000)
