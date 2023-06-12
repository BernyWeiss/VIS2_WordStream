from dataclasses import dataclass
import pandas as pd
from matplotlib.path import Path

from wordstream.util import WordStreamData

@dataclass
class Box:
    """class to represent a box in which words can be drawn.
    has information about position and size"""
    x: float
    y: float
    width: float
    height: float

def box_from_row(row) -> Box:
    """generates Box object from row where name is x and 'y', 'width' and 'height' are stored in columns"""
    return Box(row.name, row['y'], row['width'], row['height'])

def build_boxes(data: WordStreamData, width: int, height: int) -> dict[str, pd.DataFrame]:
    """Method to calculate sizes and positions of boxes based on WordStreamData
    Boxes are vertically centered and height is determined by amount of data per timeunit
    """
    tf = _total_frequencies(data)  # represents height of box
    height_scaling = height / tf.sum(axis=1).max()  # scale max frequency to height
    tf *= height_scaling
    num_boxes = len(data.df)
    box_width = width / num_boxes

    # only create index containing x-values -> y values are just frequencies
    x_index = tf.index.map(lambda i: i * box_width)
    tf = tf.set_index(x_index)

    layers = tf.cumsum(axis=1) - tf  # y starting position of boxes
    max_y = layers.iloc[:, -1].values.reshape(-1, 1)  # y starting position of top most box
    top_y = max_y + tf.iloc[:, -1].values.reshape(-1, 1)  # add height of last box to get y value for top of last box
    # center frequencies to create silhouette i.e. from (0,top_y) -> (-top_y/2,top_y/2) for each row
    layers = layers.sub(top_y / 2)

    # create dict of topics and represent boxes as dataframe with [y, width, height] as columns and x is index
    boxes = {}
    for topic in data.topics:
        topic_df = pd.DataFrame(data={"y": layers[topic]})  # x-value is index
        topic_df["height"] = tf[topic]
        topic_df["width"] = box_width
        boxes[topic] = topic_df

    return boxes


def topic_boxes_to_path(topic_boxes: pd.DataFrame) -> Path:
    """Merges boxes per topic (fraction) into a path along x-axis"""

    # top and bottom points of silhouette polygon
    top_points = []
    bottom_points = []
    for x, box in topic_boxes.iterrows():
        top_points.append((x, box.y + box.height))
        bottom_points.append((x, box.y))
        if x == topic_boxes.index.max():
            top_points.append((x + box.width, box.y + box.height))
            bottom_points.append((x + box.width, box.y))

    points = top_points + bottom_points[::-1]
    return Path(points)


def _total_frequencies(data: WordStreamData) -> pd.DataFrame:
    return data.df[data.topics].apply(lambda topic: topic.apply(lambda words: sum([w.frequency for w in words])))




