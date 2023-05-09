# Imports
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

def fill_na_and_convert_columns(colnames: list[str], df: pd.DataFrame) -> pd.DataFrame:
    for col in colnames:
        # Replace Lists with empty string with np.nan
        df[df[col] == '[""]'] = np.nan

        # Replace np.nan with empty list (formatted as string) so json.loads can be applied
        if sum(pd.isna(df[col])) > 0:
            print(f"Filling nan for column {col}.")
            df[col] = df[col].fillna('[]')
        df[col] = df[col].apply(lambda x: json.loads(x))
    return df


def get_unique_fractions(df: pd.DataFrame) -> np.ndarray:
    df = df[df['Art'] != 'BUA']

    pattern = re.compile('[a-zA-Z]+')

    fraction_as_list = list(chain.from_iterable(df["Fraktionen"]))
    parties = list(filter(lambda x: pattern.search(x), fraction_as_list))
    return np.unique(parties)


def new_get_unique_fractions(df: pd.DataFrame) -> list[str]:
    return list(set(chain.from_iterable(df["Fraktionen"])))


# Main

if __name__ == '__main__':
    load_path = "../data/" + "XX" + "/" + "antraege.csv"
    filename = "antraege.csv"

    motions_df = pd.read_csv(filepath_or_buffer=load_path, header=0, index_col=["ITYP", "INR"])
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

    motions_df = fill_na_and_convert_columns(columns_to_cleanup, motions_df)

    print(motions_df.dtypes)
    unique_parties = new_get_unique_fractions(motions_df)

    print(unique_parties)
