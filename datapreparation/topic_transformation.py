# Imports
import string
from datetime import datetime

import nltk.corpus
import pandas as pd
import numpy as np
import json
from itertools import chain
import re
from pdfminer.high_level import extract_text
import requests as rq
from bs4 import BeautifulSoup

from nltk.corpus import stopwords as sw
from nltk.tokenize import word_tokenize
# Definitions

decision_needed =  0

PARTY_MAPPING = {'V': "ÖVP",
                 'S': "SPÖ",
                 'F': "FPÖ",
                 'L': "Liberales Forum",
                 'G': "Grüne"}


ADDITIONAL_STOPWORDS = ["dr", "mag"]

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


def import_and_cleanup_csv(path, filename, legislative_period) -> pd.DataFrame:
    motions_df = pd.read_csv(filepath_or_buffer=path + filename, header=0, index_col=["ITYP", "INR"])
    motions_df["GP_CODE"] = legislative_period
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
    return motions_df


def generate_eurovoc_tsv(df, path):
    exploded_eurovoc = df[["Datum", "Fraktionen", "EUROVOC"]].explode(column="Fraktionen")
    aggregate_voc = exploded_eurovoc.groupby(["Datum", "Fraktionen"]).agg(sum)
    topics_df = aggregate_voc.reset_index().pivot(index="Datum", columns="Fraktionen", values="EUROVOC")
    for col in topics_df.columns:
        topics_df[col] = topics_df[col].apply(lambda x: [] if type(x) == float else x)
    topics_df = topics_df.applymap(lambda x: '|'.join(x))
    topics_df = topics_df.fillna("")
    topics_filename = "eurovoc.tsv"
    topics_df.to_csv(path_or_buf=path + '/' + topics_filename, sep='\t')


def generate_fulltext_tsv(df: pd.DataFrame):
    df["DocumentLinks"] = df["DocumentLinks"].fillna("[]")
    df["DocumentLinks"] = df["DocumentLinks"].apply(lambda x: x.replace("'", '"'))
    df = add_documents_datatypes(df)

    document_titles = list()

    # find different documents and their types to setup rules for choosing it
    #df.DocumentLinks.apply(lambda x: [document_titles.append(f"Typ: {doc['type']}; title: {doc['title']}") for doc in json.loads(x)])
    #print(set(document_titles))

    df["document_text"] = df.DocumentLinks.apply(lambda x: choose_document_and_return_text(json.loads(x)))

    global decision_needed
    print(f' Found {decision_needed} motions where decision was needed')
    # pdf_path = json.loads(df.DocumentLinks[0])[0]["link"]
    # html_path = json.loads(df.DocumentLinks[0])[1]["link"]
    # get_pdf_and_extract_text(pdf_path)
    # text = get_html_and_extract_text(html_path)
    # word_list = preprocess_text(text)
    print("Parsing fulltext done")


def choose_document_and_return_text(doc_links: list) -> str:
    if len(doc_links)==0:
        print("No Doc Found")
        return ""
    try:
        if len(doc_links)==1:
            if doc_links[0]["type"].lower() == "html":
                return get_html_and_extract_text(doc_links[0]["link"])
            elif doc_links[0]["type"].lower() == "pdf":
                return get_pdf_and_extract_text(doc_links[0]["link"])
            else:
                print(f"Unknown document type: {doc_links[0]}")
                return ""

        doctypes = [doc_link["type"].lower() for doc_link in doc_links]

        # Use HTML if there is one. otherwise check how many PDFs there are and find the best one
        if "html" in doctypes:
            link = doc_links[doctypes.index("html")]["link"]
            return get_html_and_extract_text(link)
        pdf_indices = [i for i, j in enumerate(doctypes) if j == 'pdf']
        if len(pdf_indices)==0:
            # no PDF found
            print(f"No PDF document found. Doctypes are: {pdf_indices}")
            return ""
        if len(pdf_indices)==1:
            return get_pdf_and_extract_text(doc_links[pdf_indices[0]]["link"])

        if len(pdf_indices)>1:
            for i in pdf_indices:
                print(f'Doctitles where there are multiple pdfs: {i}:{doc_links[i]["title"]}')
            doc_link = decide_on_doc(doc_links[pdf_indices])
    except KeyError as e:
        print(KeyError)
        return ""


def decide_on_doc(doc_links: list) -> str:
    global decision_needed
    decision_needed += 1
    return ""
    pass



def has_doctype(doc_dict_list: dict, doctype: str) -> bool:
    for doc_dict in doc_dict_list:
        if doc_dict.keys().__contains__("type"):
            if doc_dict["type"] == doctype:
                return True
    return False


def add_documents_datatypes(df: pd.DataFrame) -> pd.DataFrame:
    df["nDocs"] = df.DocumentLinks.apply(lambda x: len(json.loads(x)))
    df["hasPDF"] = df.DocumentLinks.apply(lambda x: has_doctype(json.loads(x), "PDF"))
    df["hasHTML"] = df.DocumentLinks.apply(lambda x: has_doctype(json.loads(x), "HTML"))
    df = df.assign(checkThis=lambda x: (x.hasPDF.astype(int) + x.hasHTML.astype(int)) != x.nDocs)
    print(f"There are {sum(df.checkThis)} out of {len(df.checkThis)} with multiple documents in a type")
    return df




def get_html_and_extract_text(relative_link: str) -> str:
    base_link = "https://www.parlament.gv.at"

    response = rq.get(base_link + relative_link)
    html_doc = response.content
    soup = BeautifulSoup(html_doc, features="html.parser")
    # kill all script and style elements
    for script in soup(["script", "style"]):
        script.extract()  # rip it out

    # get text
    text = soup.get_text()
    return text


def get_pdf_and_extract_text(relative_link: str) -> str:
    base_link = "https://www.parlament.gv.at"

    pdf = rq.get(base_link + relative_link)

    with open('example.pdf', 'wb') as f:
        f.write(pdf.content)

    text = extract_text("example.pdf")

    return text

def preprocess_text(text: str) -> list[str]:
    tokenized_text = word_tokenize(text, "german")

    lower_text = [word.casefold() for word in tokenized_text]

    stopwords = sw.words("german")

    # Adds more words to stopwordlist.
    stopwords.extend(ADDITIONAL_STOPWORDS)
    clean_text = [word for word in lower_text if not word in stopwords and word not in string.punctuation]

    return clean_text


# Main


if __name__ == '__main__':
    legislative_period = "XX"
    path = "../data/" + legislative_period + "/"
    filename = "antraege.csv"

    clean_df = import_and_cleanup_csv(path, filename, legislative_period)

    only_fractions_and_documents = clean_df.loc[:, ["Datum", "Fraktionen", "DocumentLinks"]]

    generate_fulltext_tsv(only_fractions_and_documents)


    # unique_parties = get_unique_fractions(clean_df)
    # unique_dates = np.unique(clean_df["Datum"])

    # use this to generate topic files based on eurovoc
    # generate_eurovoc_tsv(clean_df, path)

    # print(unique_parties)
