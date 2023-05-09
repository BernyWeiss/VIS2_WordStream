# Imports
import pandas as pd
import numpy as np
from itertools import chain


# Functions

def get_unique_fractions(df: pd.DataFrame) -> list[str]:
    df = df[df['ITYP'] != 'I']
    parties = np.unique(list(chain.from_iterable(df["Fraktionen"])))
    return parties

# Main

if __name__ == '__main__':
    load_path = "../data/" + "XXVII" + "/" + "antraege.csv"
    filename = "antraege.csv"

    motions_df = pd.read_csv(filepath_or_buffer=load_path, header=0, index_col="HIS_URL")
    parties = get_unique_fractions(motions_df)

    print(parties)

