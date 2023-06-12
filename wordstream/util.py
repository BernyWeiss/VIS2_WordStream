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

GP_TIME_INTERVALS = {"XX": (datetime(year=1996, month=1, day=15), datetime(year=1999, month=7, day=16)),
                     "XXI": (datetime(year=1999, month=10, day=29), datetime(year=2002, month=9, day=20)),
                     "XXII": (datetime(year=2002, month=12, day=20), datetime(year=2006, month=9, day=21)),
                     "XXIII": (datetime(year=2006, month=10, day=30), datetime(year=2008, month=10, day=20)),
                     "XXIV": (datetime(year=2008, month=10, day=28), datetime(year=2013, month=9, day=25)),
                     "XXV": (datetime(year=2013, month=10, day=29), datetime(year=2017, month=10, day=13)),
                     "XXVI": (datetime(year=2017, month=11, day=9), datetime(year=2019, month=9, day=26)),
                     "XXVII": (datetime(year=2019, month=10, day=23), datetime(year=2023, month=6, day=1))}

# Classes
@dataclass
class WordStreamData:
    df: pd.DataFrame
    topics: list[str]
    segment_start: dict[str, datetime]


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
    df = pd.read_csv(path, sep="\t", header=0, parse_dates=[time_col], keep_default_na=False).drop(columns=list(drop))
    df = df.groupby(time_col).apply(lambda col: col.apply("|".join)).reset_index(names=[time_col])
    topics = [c for c in df.columns if c != time_col]
    df[topics] = df[topics].apply(lambda col: col.apply(lambda v: v.split("|")))
    return WordStreamData(df=df.sort_values(time_col, ascending=True).reset_index(drop=True), topics=topics)

def load_multiple_periods(path: str, filename: str, periods: list[str], time_col: str = "time", drop: tuple[str] = ("source",)) -> WordStreamData:
    df = pd.DataFrame()
    period_startdates = {}
    for period in periods:
        temp = pd.read_csv(path +"/"+period+"/"+filename, sep="\t", header=0, parse_dates=[time_col], keep_default_na=False).drop(columns=list(drop))
        period_startdates[period] = temp.at[0,time_col]
        df = pd.concat([df, temp])
    df = df.fillna("")
    df = df.groupby(time_col).apply(lambda col: col.apply("|".join)).reset_index(names=[time_col])
    topics = [c for c in df.columns if c != time_col]
    df[topics] = df[topics].apply(lambda col: col.apply(lambda v: v.split("|")))
    return WordStreamData(df=df.sort_values(time_col, ascending=True).reset_index(drop=True),
                          topics=topics,
                          segment_start=period_startdates)

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
    data = load_data("../data/FactCheck.tsv")
    data.df = group_by_date(data.df)
    data.df = data.df[(data.df.time >= start) & (data.df.time <= end)].reset_index(drop=True)
    data.df = calculate_word_frequency(data.df, data.topics)
    data.df = calculate_sudden(data.df, data.topics, top=50)
    return data

def load_parlament_data(periods: list[str], fulltext: bool = False)-> WordStreamData:
    filename = "eurovoc.tsv"
    if fulltext:
        filename = "fulltext.tsv"

    time_col = "Datum"
    start = GP_TIME_INTERVALS[periods[0]][0]
    end = GP_TIME_INTERVALS[periods[len(periods)-1]][1]
    data = load_multiple_periods("../data/", filename, periods, time_col, ())
    data.df = group_by_date(data.df, group_col=time_col, freq="Y")
    data.df = data.df[(data.df[time_col] >= start) & (data.df[time_col] <= end)].reset_index(drop=True)
    data.df = calculate_word_frequency(data.df, data.topics)
    data.df = calculate_sudden(data.df, data.topics, top=50)

    # change order of columns to improve box placements
    col_names = data.topics
    first_cols = ["F","S","V"]
    other_cols = [name for name in col_names if name not in first_cols]
    first_cols.extend(other_cols)
    data.topics = first_cols
    sorted_colums = []
    sorted_colums.extend(data.topics)
    sorted_colums.append(time_col)
    data.df = data.df[sorted_colums]
    return data



def get_max_sudden(data: WordStreamData):
    topics_df = data.df[data.topics]
    return topics_df.apply(
        lambda col: col.apply(
            lambda l: max(l, key=lambda w: w.sudden) if len(l)>0 else 0
        )
    ).max(axis=None).sudden
