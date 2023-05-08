# Imports
from enum import Enum
from util import ROMAN_NUMERAL, roman_to_int, int_to_roman
from dataclasses import dataclass, field
import requests as rq
import json
import pandas as pd

# Definitions
MIN_LEGISLATIVE_PERIOD = 5
MAX_LEGISLATIVE_PERIOD = 27


# Classes
class LegislativeBody(Enum):
    NATIONAL_ASSEMBLY = "NR"
    FEDERAL_ASSEMBLY = "BR"


@dataclass
class RomanNumeral:
    def __init__(self, decimal):
        decimal: int
        roman: str = field(init=False)

    def __post_init(self):
        self.roman = int_to_roman(self.decimal)

    # Functions


def add_value_to_request_body(req_body: dict, body_key: str, body_value) -> dict:
    to_add = []
    if isinstance(body_value, str):
        to_add.append(body_value)
    else:
        to_add.extend(body_value)

    if body_key not in req_body.keys():
        req_body[body_key] = []
    req_body[body_key].extend(to_add)

    return req_body


def get_motions(periods: list[int], legis_bodies: list[LegislativeBody]):
    roman_periods = [RomanNumeral(x) for x in periods]

    request_body = {}

    request_body = add_value_to_request_body(request_body, "NRBR", "NR")
    request_body = add_value_to_request_body(request_body, "GP_CODE", "XXVII")
    request_body = add_value_to_request_body(request_body, "VHG", "ANTR")

    res = rq.post(
        "https://www.parlament.gv.at/Filter/api/filter/data/101?js=eval&showAll=true&export=true",
        data=json.dumps(request_body))

    return res


def parse_gegenstand_document_links(detail_json):
    return detail_json['content']['documents']


def add_doc_links_to_df(motion_df: pd.DataFrame) -> pd.DataFrame:
    # TODO: Finish this
    motion_ids = motion_df.index.array

    for motion_id in motion_ids:
        res = rq.get("https://www.parlament.gv.at" + motion_id + "?json=true")
        doc_links = parse_gegenstand_document_links(res.json())
        pass
    return motion_df

# TODO: Add CSV Export

# Main

if __name__ == '__main__':
    response = get_motions([1, 2, 3], [])

    res_json = response.json()

    col_labels = [header_entry["label"] for header_entry in res_json["header"]]

    motion_df = pd.DataFrame(data=res_json["rows"], columns=col_labels)
    motion_df.set_index("HIS_URL", inplace=True)
    cols_to_drop = ["INR", "INRNUM", "DATUMSORT", "NR_GP_CODE",
                    "DATUM_VON", "Nummer", "PHASEN_BIS", "Personen", "RSS_DESC", "ZUKZ",
                    "Status", "WENTRY_ID", "Zust?"]
    motion_df = motion_df.drop(cols_to_drop, axis=1)
    motion_df = add_doc_links_to_df(motion_df)

    print(response.json())
