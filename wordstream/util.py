# Imports
from functools import total_ordering

import pandas as pd
import math

from collections.abc import Sequence
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from operator import add


# Definitions

# Classes
@dataclass
class WordStreamData:
    df: pd.DataFrame
    topics: list[str]


@dataclass
@total_ordering
class Word:
    text: str
    frequency: int
    sudden: float

    def __eq__(self, other):
        return isinstance(other, Word) and self.sudden == other.sudden

    def __lt__(self, other):
        return isinstance(other, Word) and self.sudden < other.sudden


# Functions
def int_to_roman(n: int) -> str:
    result = ''
    allSymbol = ['M', 'CM', 'D', "CD", 'C', 'XC', 'L', 'XL', 'X', 'IX', 'V', 'IV', 'I']
    value = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
    for index in range(len(value)):
        quotient = n // value[index]  # to know the number of times the symbol should be written
        symbol = quotient * allSymbol[index]  # to write the symbol in specific times
        result += symbol  # this is to add the symbol to the result.
        n = n % value[index]  # this is to get the remainder which will be use again
    return result


def load_data(path: str, time_col: str = "time", drop: tuple[str] = ("source",)) -> WordStreamData:
    df = pd.read_csv(path, sep="\t", header=0, parse_dates=[time_col]).drop(columns=list(drop))
    df = df.groupby(time_col).apply(lambda col: col.apply("|".join)).reset_index(names=["time"])
    topics = [c for c in df.columns if c != time_col]
    df[topics] = df[topics].apply(lambda col: col.apply(lambda v: v.split("|")))
    return WordStreamData(df=df.sort_values("time", ascending=True).reset_index(drop=True), topics=topics)


def calculate_word_frequency(df: pd.DataFrame, topics: list[str]) -> pd.DataFrame:
    df[topics] = df[topics].apply(lambda col: col.apply(lambda words: Counter(words)))
    return df


def _counter_to_list(c: Counter, prev_c: Counter, top: int | None = None) -> list[Word]:
    return sorted([
        Word(text=word, frequency=count, sudden=(count + 1) / (prev_c[word] + 1))
        for word, count in c.most_common(n=top) if word
    ], key=lambda k: k.sudden, reverse=True)


def calculate_sudden(df: pd.DataFrame, topics: list[str], top: int | None = None) -> pd.DataFrame:
    def sudden_from_idx(i: int, topic: pd.Series) -> list[Word]:
        if i == 0:
            return _counter_to_list(topic.iloc[i], Counter(), top)
        else:
            return _counter_to_list(topic.iloc[i], topic.iloc[i - 1], top)

    df[topics] = df[topics].apply(lambda col: col.index.map(lambda i: sudden_from_idx(i, col)))
    return df


def group_by_date(df: pd.DataFrame, freq: str = "MS", group_col: str = "time"):
    df_grouped = df.groupby(pd.Grouper(key=group_col, freq=freq))
    df_concat = df_grouped.agg(lambda group: [e for l in group for e in l]).reset_index(names=[group_col])
    return df_concat


def load_fact_check(start: datetime = datetime(year=2013, month=1, day=1), end=datetime(year=2013, month=12, day=31)) -> WordStreamData:
    data = load_data("../../data/FactCheck.tsv")
    data.df = group_by_date(data.df)
    data.df = data.df[(data.df.time >= start) & (data.df.time <= end)].reset_index(drop=True)
    data.df = calculate_word_frequency(data.df, data.topics)
    data.df = calculate_sudden(data.df, data.topics, top=15)
    return data


def get_max_sudden(data: WordStreamData):
    topics_df = data.df[data.topics]
    return topics_df.apply(
        lambda col: col.apply(
            lambda l: max(l, key=lambda w: w.sudden)
        )
    ).max(axis=None).sudden
