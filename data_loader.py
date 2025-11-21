import requests
import pandas as pd
from typing import Dict

AVANZA_BASE_URL = (
    "https://www.avanza.se/_api/fund-guide/chart/"
    "{fund_id}/{from_date}/{to_date}?raw=true"
)


def fetch_fund_history(fund_id: str, from_date: str, to_date: str) -> pd.Series:
    """
    Hämtar fondhistorik från Avanza och returnerar en prisserie
    (pd.Series med datumindex).
    """
    url = AVANZA_BASE_URL.format(
        fund_id=fund_id,
        from_date=from_date,
        to_date=to_date,
    )
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()

    serie = pd.DataFrame(data["dataSerie"])
    serie["date"] = pd.to_datetime(serie["x"], unit="ms")
    serie = serie.set_index("date")["y"].astype(float)
    serie.name = f'{data.get("name", "Fond")} ({fund_id})'
    return serie


def build_return_matrix(price_series: Dict[str, pd.Series], freq: str = "ME") -> pd.DataFrame:
    """
    Bygger synkroniserade logiska avkastningar från prisserier.
    freq="ME" = month-end (ersätter deprecated "M").
    """
    dfs = []
    for name, s in price_series.items():
        r = s.resample(freq).last().pct_change()
        r.name = name
        dfs.append(r)
    returns = pd.concat(dfs, axis=1).dropna()
    return returns
