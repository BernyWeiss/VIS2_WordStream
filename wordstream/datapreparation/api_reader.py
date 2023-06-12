"""
Script file to read motions from the Open Data API of the Austrian Parliament
"""

# Imports
from enum import Enum
from collections.abc import Sequence
import requests as rq
import json
import pandas as pd
import os
from tqdm import tqdm

# Definitions
MIN_LEGISLATIVE_PERIOD = 5
MAX_LEGISLATIVE_PERIOD = 27


# Classes
class LegislativeBody(Enum):
    """Possible Codes for legislative bodies to request."""
    NATIONAL_ASSEMBLY = "NR"
    FEDERAL_ASSEMBLY = "BR"


# Functions
def _int_to_roman(n: int) -> str:
    result = ''
    allSymbol = ['M', 'CM', 'D', "CD", 'C', 'XC', 'L', 'XL', 'X', 'IX', 'V', 'IV', 'I']
    value = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
    for index in range(len(value)):
        quotient = n // value[index]  # to know the number of times the symbol should be written
        symbol = quotient * allSymbol[index]  # to write the symbol in specific times
        result += symbol  # this is to add the symbol to the result.
        n = n % value[index]  # this is to get the remainder which will be use again
    return result

def _add_value_to_request_body(req_body: dict, body_key: str, body_value) -> dict:
    to_add = []
    if isinstance(body_value, str):
        to_add.append(body_value)
    else:
        to_add.extend(body_value)

    if body_key not in req_body.keys():
        req_body[body_key] = []
    req_body[body_key].extend(to_add)

    return req_body



def get_motions(periods: list[str], legis_bodies: Sequence[LegislativeBody]):
    """Requests the motions from parlament.gv.at.

    Keyword arguments:
    periods -- list of legislative periods as roman numbers
    legis_bodies -- abbreviation of legislative body
    """
    request_body = {}

    request_body = _add_value_to_request_body(request_body, "NRBR", legis_bodies)
    request_body = _add_value_to_request_body(request_body, "GP_CODE", periods)
    request_body = _add_value_to_request_body(request_body, "VHG", "ANTR")

    res = rq.post(
        "https://www.parlament.gv.at/Filter/api/filter/data/101?js=eval&showAll=true&export=true",
        data=json.dumps(request_body))

    return res



def parse_gegenstand_document_links(detail_json: dict) -> list[dict]:
    """Parses json of 'Gegenstand' to extract the title, link and type of documents attached to it"""
    if 'documents' not in detail_json['content'].keys():
        return []

    docs = detail_json['content']['documents']
    links = []
    for document_group in docs:
        if "title" in document_group.keys() and "documents" in document_group.keys():
            for document in document_group["documents"]:
                links.append({'title': document_group["title"],
                              'link': document["link"],
                              'type': document["type"]})

    return links


def parse_document_links(df: pd.DataFrame) -> pd.Series:
    """
    Requests 'Gegenstand' detail-site and parses document links for DataFrame of 'Gegenstand'
    """
    motion_index = df.index

    document_links = pd.Series(index=motion_index, dtype=str)

    for leg_period, ityp, inr in tqdm(motion_index):
        res = rq.get(f"https://www.parlament.gv.at/gegenstand/{leg_period}/{ityp}/{inr}?json=true")
        document_links.at[(leg_period,ityp,inr)] = str(parse_gegenstand_document_links(res.json()))
    return document_links


def export_df_to_csv(path: str, df: pd.DataFrame) -> None:
    """Exports DataFrame of Motions to CSV file in given directory"""
    filename = "antraege.csv"
    os.makedirs(path, exist_ok=True)

    df.to_csv(path_or_buf=path + '/' + filename)


# Main
if __name__ == '__main__':
    """Script to request motions and save them to csv"""
    legis_periods = [22,23,24,25,26,27]
    roman_periods = [_int_to_roman(n) for n in legis_periods]

    for period in roman_periods:
        print(f"Start requesting Data for Legis Period {period}.")
        response = get_motions(period, "NR")

        res_json = response.json()

        col_labels = [header_entry["label"] for header_entry in res_json["header"]]

        motion_df = pd.DataFrame(data=res_json["rows"], columns=col_labels)
        motion_df.set_index(["GP_CODE", "ITYP", "INR"], inplace=True)
        cols_to_drop = ["DATUMSORT", "NR_GP_CODE", "LZ-Buttons",
                        "DATUM_VON", "Nummer", "PHASEN_BIS", "Personen", "RSS_DESC", "ZUKZ",
                        "Status", "WENTRY_ID", "Zust?", "sysdate???", "INRNUM", "NRBR"]
        motion_df = motion_df.drop(cols_to_drop, axis=1)

        print(f"Start retrieving document links for Legis Period {period}")
        motion_df['DocumentLinks'] = parse_document_links(motion_df)

        print("Retrieving document links done! Start exporting to CSV")
        save_path = "../data/" + period + "/"
        export_df_to_csv(save_path, motion_df.loc[period])
        print(f"Export for period {period} done.")
