# Imports
import pandas as pd
import math

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from operator import add

# Definitions
ROMAN_NUMERAL = {'M': 1000, 'D': 500, 'C': 100, 'L': 50, 'X': 10, 'V': 5, 'I': 1}


# Classes
@dataclass
class WordStreamData:
    df: pd.DataFrame
    topics: list[str]


@dataclass
class Word:
    text: str
    frequency: int
    sudden: float


# Functions
def int_to_roman(n: int):
    div = 1
    while n >= div:
        div *= 10
    div /= 10
    res = ""
    while n:
        # main significant digit extracted
        # into lastNum
        last_num = int(n / div)
        if last_num <= 3:
            res += (ROMAN_NUMERAL[div] * last_num)
        elif last_num == 4:
            res += (ROMAN_NUMERAL[div] +
                    ROMAN_NUMERAL[div * 5])
        elif 5 <= last_num <= 8:
            res += (ROMAN_NUMERAL[div * 5] +
                    (ROMAN_NUMERAL[div] * (last_num - 5)))
        elif last_num == 9:
            res += (ROMAN_NUMERAL[div] +
                    ROMAN_NUMERAL[div * 10])
        n = math.floor(n % div)
        div /= 10
    return res


def roman_to_int(s: str):
    res, p = 0, 'I'
    for c in s[::-1]:
        res, p = res - ROMAN_NUMERAL[c] if ROMAN_NUMERAL[c] < ROMAN_NUMERAL[p] else res + ROMAN_NUMERAL[c], c
    return res


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


def load_fact_check(top: int | None = None, start: datetime = datetime(year=2012, month=1, day=1)) -> WordStreamData:
    data = load_data("../../data/FactCheck.tsv")
    data.df = data.df[data.df.time >= start].reset_index(drop=True)
    data.df = calculate_word_frequency(data.df, data.topics)
    data.df = calculate_sudden(data.df, data.topics, top)
    return data
