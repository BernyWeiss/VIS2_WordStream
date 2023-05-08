import requests as rq
from sqlalchemy import create_engine, ForeignKey
from sqlalchemy.orm import DeclarativeBase, mapped_column, Mapped, Session, relationship
from tqdm import tqdm
from typing import List


class Base(DeclarativeBase):
    pass


class Mandat(Base):
    __tablename__ = "mandat"
    id: Mapped[int] = mapped_column(primary_key=True)
    mandatar_id: Mapped[int] = mapped_column(ForeignKey("person.id"))
    mandat: Mapped[str]
    klub: Mapped[str] = mapped_column(nullable=True)
    wahlkreis: Mapped[str] = mapped_column(nullable=True)
    wahlpartei: Mapped[str] = mapped_column(nullable=True)
    wahlpartei_text: Mapped[str] = mapped_column(nullable=True)
    gremium: Mapped[str]
    bez: Mapped[str]
    mandatVon: Mapped[str]
    mandatBis: Mapped[str] = mapped_column(nullable=True)
    aktiv: Mapped[bool]
    zeitraum: Mapped[str]
    eingetreten_txt: Mapped[str] = mapped_column(nullable=True)


class Person(Base):
    __tablename__ = "person"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str]
    sex: Mapped[str]
    mandate: Mapped[List["Mandat"]] = relationship()


def parse_person(row, detail):
    p = Person()
    p.id = row[2]
    p.name = row[3]
    p.sex = row[5]
    for m in detail['content']['documents']['mandatefunktionen']['mandate']:
        p.mandate.append(Mandat(**m))
    return p


def get_persons(engine):
    res = rq.post(
        "https://www.parlament.gv.at/Filter/api/filter/data/10400?js=eval&showAll=true&export=true&pagesize=10000",
        json={})
    persons = []
    for p in tqdm(res.json()['rows']):
        if p[4]:
            res = rq.get("https://www.parlament.gv.at" + p[4] + "?json=true")
            persons.append(parse_person(p, res.json()))
    with Session(engine) as session:
        for p in persons:
            session.merge(p)
        session.commit()



if __name__ == '__main__':
    engine = create_engine("sqlite+pysqlite:///db")
    Base.metadata.create_all(engine)

    get_persons(engine)
