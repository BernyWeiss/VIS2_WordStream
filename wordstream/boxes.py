import pandas as pd

from wordstream.util import WordStreamData, load_fact_check


def build_boxes(data: WordStreamData, width: int) -> dict[str, pd.DataFrame]:
    tf = total_frequencies(data)  # represents height of box
    num_boxes = len(data.df)
    box_width = width // num_boxes

    # only create index containing x-values -> y values are just frequencies
    x_index = tf.index.map(lambda i: (i * box_width) + (box_width >> 1))
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


def total_frequencies(data: WordStreamData) -> pd.DataFrame:
    return data.df[data.topics].apply(lambda topic: topic.apply(lambda words: sum([w.frequency for w in words])))


if __name__ == '__main__':
    data = load_fact_check()
    build_boxes(data, 1000)
