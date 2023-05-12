# Imports
from datetime import datetime

import pandas as pd
import numpy as np
import json
from itertools import chain
import re


# Definitions

PARTY_MAPPING = {'V': "ÖVP",
                 'S': "SPÖ",
                 'F': "FPÖ",
                 'L': "Liberales Forum",
                 'G': "Grüne"}


# Functions

def fill_na_and_convert_columns_to_object_types(colnames: list[str], df: pd.DataFrame) -> pd.DataFrame:
    for col in colnames:
        # Replace Lists with empty string with np.nan
        df[df[col] == '[""]'] = np.nan

        # Replace np.nan with empty list (formatted as string) so json.loads can be applied
        if sum(pd.isna(df[col])) > 0:
            print(f"Filling nan for column {col}.")
            df[col] = df[col].fillna('[]')
        df[col] = df[col].apply(lambda x: json.loads(x))
    return df


def get_unique_fractions(df: pd.DataFrame) -> list[str]:
    return list(set(chain.from_iterable(df["Fraktionen"])))


# Main

if __name__ == '__main__':
    path = "../data/" + "XX" + "/"
    filename = "antraege.csv"

    motions_df = pd.read_csv(filepath_or_buffer=path + filename, header=0, index_col=["ITYP", "INR"])
    motions_df["GP_CODE"] = "XX"
    motions_df = motions_df.set_index("GP_CODE", append=True)
    motions_df = motions_df.reorder_levels([2, 0, 1])
    motions_df = motions_df.astype({'Art': str,
                                    'Betreff': str,
                                    'DOKTYP': str,
                                    'DOKTYP_LANG': str,
                                    'HIS_URL': str,
                                    'VHG': str,
                                    'VHG2': str})
    motions_df['Datum'] = pd.to_datetime(motions_df['Datum'], format='%d.%m.%Y')

    columns_to_cleanup = ["Fraktionen", "THEMEN", "SW", "EUROVOC"]

    motions_df = fill_na_and_convert_columns_to_object_types(columns_to_cleanup, motions_df)

    unique_parties = get_unique_fractions(motions_df)

    # topics_df = pd.DataFrame(index=pd.Series(data=None,
    #                                         name="time",
    #                                         dtype=datetime),
    #                         columns=unique_parties)

    unique_dates = np.unique(motions_df["Datum"])

    exploded_eurovoc = motions_df[["Datum", "Fraktionen", "EUROVOC"]].explode(column="Fraktionen")
    aggregate_voc = exploded_eurovoc.groupby(["Datum", "Fraktionen"]).agg(sum)

    topics_df = aggregate_voc.reset_index().pivot(index="Datum", columns="Fraktionen", values="EUROVOC")

    for col in topics_df.columns:
        topics_df[col] = topics_df[col].apply(lambda x: [] if type(x) == float else x)

    topics_df = topics_df.applymap(lambda x: '|'.join(x))
    topics_df = topics_df.fillna("")

    topics_filename = "eurovoc.tsv"
    topics_df.to_csv(path_or_buf=path + '/' + topics_filename, sep='\t')

    print(unique_parties)
