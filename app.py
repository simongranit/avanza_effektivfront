import datetime as dt
from typing import Dict, List

import numpy as np
import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt
import json

AVANZA_BASE_URL = (
    "https://www.avanza.se/_api/fund-guide/chart/"
    "{fund_id}/{from_date}/{to_date}?raw=true"
)

# ------------------------------
# Hjälpfunktioner
# ------------------------------

@st.cache_data(show_spinner=False)
def fetch_fund_history(fund_id: str, from_date: str, to_date: str) -> pd.Series:
    """Hämtar fondhistorik från Avanza och returnerar prisserie (Series med datumindex)."""
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


def build_return_matrix(price_series: Dict[str, pd.Series], freq: str = "M") -> pd.DataFrame:
    """Bygger synkroniserade månadsavkastningar från prisserier."""
    dfs = []
    for name, s in price_series.items():
        r = s.resample(freq).last().pct_change()
        r.name = name
        dfs.append(r)
    returns = pd.concat(dfs, axis=1).dropna()
    return returns


def sample_weights_with_constraints(
    n_assets: int,
    swe_eq_idx: np.ndarray,
    for_eq_idx: np.ndarray,
    bond_idx: np.ndarray,
    max_weight: float,
    max_equity: float,
    max_foreign_equity_frac: float,
    min_equity: float = 0.0,
    max_tries: int = 3000,
) -> np.ndarray:
    """
    Slumpar vikter med:
      - sum(w) = 1
      - 0 <= w_i <= max_weight
      - min_equity <= sum(equity) <= max_equity
      - (sum(foreign_equity) / sum(equity)) <= max_foreign_equity_frac
    """
    n_se = len(swe_eq_idx)
    n_fe = len(for_eq_idx)
    n_b = len(bond_idx)

    for _ in range(max_tries):
        w = np.zeros(n_assets)

        # 1) Slumpa total aktieandel
        if (n_se + n_fe) > 0:
            equity_share = np.random.uniform(min_equity, max_equity)
        else:
            equity_share = 0.0
        bond_share = 1.0 - equity_share

        # per-fond-cap grovkoll
        if (n_se + n_fe) > 0 and (n_se + n_fe) * max_weight < equity_share - 1e-9:
            continue
        if n_b > 0 and n_b * max_weight < bond_share - 1e-9:
            continue
        if n_b == 0 and bond_share > 1e-9:
            # kräver räntedel men har inga räntefonder
            continue

        # 2) Dela upp aktiedelen i svensk & utländsk med constraint
        if equity_share > 0 and n_fe > 0 and max_foreign_equity_frac > 0:
            # andel av aktiedelen som får vara utländsk
            u = np.random.uniform(0.0, max_foreign_equity_frac)
            foreign_share = equity_share * u
        else:
            foreign_share = 0.0

        swe_share = equity_share - foreign_share

        # feasibility per-fond inom grupper
        if n_fe > 0 and n_fe * max_weight < foreign_share - 1e-9:
            continue
        if n_se > 0 and n_se * max_weight < swe_share - 1e-9:
            continue
        if n_se == 0 and swe_share > 1e-9:
            # ingen svensk aktiefond men krav på svensk del
            continue

        # 3) Slumpa inom svensk aktiedel
        if n_se > 0 and swe_share > 0:
            we_s = np.random.dirichlet(np.ones(n_se)) * swe_share
            for _ in range(20):
                over = we_s > max_weight + 1e-9
                if not over.any():
                    break
                excess = we_s[over] - max_weight
                we_s[over] = max_weight
                under = ~over
                if under.any():
                    we_s[under] += excess.sum() * (we_s[under] / we_s[under].sum())
                else:
                    break
        else:
            we_s = np.zeros(n_se)

        # 4) Slumpa inom utländsk aktiedel
        if n_fe > 0 and foreign_share > 0:
            we_f = np.random.dirichlet(np.ones(n_fe)) * foreign_share
            for _ in range(20):
                over = we_f > max_weight + 1e-9
                if not over.any():
                    break
                excess = we_f[over] - max_weight
                we_f[over] = max_weight
                under = ~over
                if under.any():
                    we_f[under] += excess.sum() * (we_f[under] / we_f[under].sum())
                else:
                    break
        else:
            we_f = np.zeros(n_fe)

        # 5) Slumpa inom räntedel
        if n_b > 0 and bond_share > 0:
            wb = np.random.dirichlet(np.ones(n_b)) * bond_share
            for _ in range(20):
                over = wb > max_weight + 1e-9
                if not over.any():
                    break
                excess = wb[over] - max_weight
                wb[over] = max_weight
                under = ~over
                if under.any():
                    wb[under] += excess.sum() * (wb[under] / wb[under].sum())
                else:
                    break
        else:
            wb = np.zeros(n_b)

        w[swe_eq_idx] = we_s
        w[for_eq_idx] = we_f
        w[bond_idx] = wb

        # slutkoller
        if not np.isclose(w.sum(), 1.0, atol=1e-6):
            continue
        if (w > max_weight + 1e-6).any():
            continue

        total_equity = w[swe_eq_idx].sum() + w[for_eq_idx].sum()
        foreign_equity = w[for_eq_idx].sum()

        if total_equity > 0:
            frac_foreign_equity = foreign_equity / total_equity
            if frac_foreign_equity > max_foreign_equity_frac + 1e-6:
                continue

        if not (min_equity - 1e-6 <= total_equity <= max_equity + 1e-6):
            continue

        return w

    raise RuntimeError("Kunde inte generera vikter med givna restriktioner (inkl. utländsk aktiedel).")


def simulate_portfolios(
    returns: pd.DataFrame,
    swe_eq_mask: np.ndarray,
    for_eq_mask: np.ndarray,
    max_weight: float,
    max_equity_share: float,
    max_foreign_equity_frac: float,
    n_portfolios: int,
    rf: float,
):
    """Monte Carlo-simulering med alla constraints."""
    mean_returns = returns.mean() * 12
    cov_matrix = returns.cov() * 12
    n_assets = returns.shape[1]

    swe_idx = np.where(swe_eq_mask)[0]
    for_idx = np.where(for_eq_mask)[0]
    bond_idx = np.where(~(swe_eq_mask | for_eq_mask))[0]

    port_returns, port_vols, sharpes, weights = [], [], [], []

    for _ in range(n_portfolios):
        w = sample_weights_with_constraints(
            n_assets=n_assets,
            swe_eq_idx=swe_idx,
            for_eq_idx=for_idx,
            bond_idx=bond_idx,
            max_weight=max_weight,
            max_equity=max_equity_share,
            max_foreign_equity_frac=max_foreign_equity_frac,
        )
        r = np.dot(w, mean_returns)
        v = np.sqrt(w.T @ cov_matrix.values @ w)
        s = (r - rf) / v if v > 0 else np.nan

        port_returns.append(r)
        port_vols.append(v)
        sharpes.append(s)
        weights.append(w)

    return {
        "returns": np.array(port_returns),
        "vols": np.array(port_vols),
        "sharpes": np.array(sharpes),
        "weights": np.array(weights),
        "mean_returns": mean_returns,
        "cov_matrix": cov_matrix,
    }



def describe_portfolio(
    weights: np.ndarray,
    asset_names: List[str],
    annual_returns: pd.Series,
    cov_matrix: pd.DataFrame,
) -> pd.Series:
    """Skapar en Serie med avkastning, risk och vikter."""
    w = weights
    r = np.dot(w, annual_returns)
    v = np.sqrt(w.T @ cov_matrix.values @ w)
    s = pd.Series({"Förväntad årsavkastning": r, "Årsvolatilitet": v})
    for i, name in enumerate(asset_names):
        s[f"Vikt {name}"] = w[i]
    return s


# ------------------------------
# Streamlit layout
# ------------------------------

st.set_page_config(page_title="Portföljsimulator (Avanza)", layout="wide")
st.title("📊 Portföljsimulator med Avanza-fonder")

st.markdown(
    """
Flöde:

1. **Ladda fonder** via Avanza-ID  
2. **Välj & klassificera** (räntefond / svensk aktiefond / utländsk aktiefond)  
3. **Kör simulering** med begränsningar:
   - max vikt per fond  
   - max andel aktiefonder totalt  
   - max andel **utländska aktiefonder av aktiedelen**
"""
)

# init state
if "price_series" not in st.session_state:
    st.session_state["price_series"] = None
if "fund_types" not in st.session_state:
    st.session_state["fund_types"] = {}

# SIDEBAR
st.sidebar.header("Generella inställningar")

st.sidebar.markdown("### Ladda sparad konfiguration")
config_file = st.sidebar.file_uploader("Välj JSON-fil", type=["json"])

if config_file is not None:
    try:
        loaded_config = json.load(config_file)
        st.session_state["loaded_config"] = loaded_config

        ids_from_config = sorted({str(item["fund_id"]) for item in loaded_config if item.get("fund_id")})
        st.sidebar.success(f"Laddade {len(ids_from_config)} fonder från konfiguration.")
        st.sidebar.write("Fonder i filen:", ", ".join(ids_from_config))
    except Exception as e:
        st.sidebar.error(f"Kunde inte läsa konfigurationen: {e}")


ids_input = st.sidebar.text_area(
    "Avanza fond-ID (komma- eller radseparerade)",
    value="2111\n1961\n315116\n2014\n1971\n652895\n592236\n2069\n2066\n694253\n363\n685395\n1505\n325406\n878733\n536695\n2088",
)

today = dt.date.today()
start_date = st.sidebar.date_input("Startdatum", dt.date(2005, 1, 1))
end_date = st.sidebar.date_input("Slutdatum", today)

rf_input = st.sidebar.number_input("Riskfri ränta (år, %)", 0.0, 20.0, 2.0, step=0.25)
rf = rf_input / 100.0

max_weight_input = st.sidebar.number_input("Max vikt per fond (%)", 1.0, 100.0, 15.0)
max_weight = max_weight_input / 100.0

max_equity_input = st.sidebar.number_input(
    "Max andel aktiefonder (%)", 0.0, 100.0, 75.0, step=5.0
)
max_equity_share = max_equity_input / 100.0

max_foreign_equity_input = st.sidebar.number_input(
    "Max andel utländska aktiefonder av aktiedelen (%)",
    0.0,
    100.0,
    25.0,
    step=5.0,
)
max_foreign_equity_frac = max_foreign_equity_input / 100.0

min_display_weight_input = st.sidebar.number_input(
    "Min vikt att visa per fond (%)",
    0.0,
    100.0,
    5.0,
    step=1.0,
)
min_display_weight = min_display_weight_input / 100.0


n_portfolios = int(
    st.sidebar.number_input("Antal simulerade portföljer", 1000, 50000, 5000, step=1000)
)

# ---- STEG 1: Ladda fonder ----
st.subheader("Steg 1: Ladda fonder från Avanza")
load_button = st.button("🔄 Ladda fonder")

if load_button:
    ids = [x.strip() for x in ids_input.replace(",", "\n").splitlines() if x.strip()]
    if not ids:
        st.error("Du måste ange minst ett Avanza-ID.")
    else:
        price_series: Dict[str, pd.Series] = {}
        errors = []

        from_str = start_date.isoformat()
        to_str = end_date.isoformat()

        progress = st.progress(0.0)
        for i, fid in enumerate(ids, start=1):
            try:
                s = fetch_fund_history(fid, from_str, to_str)
                price_series[s.name] = s
            except Exception as e:
                errors.append((fid, str(e)))
            progress.progress(i / len(ids))

        if errors:
            st.warning("Kunde inte läsa in vissa fonder:")
            for fid, msg in errors:
                st.write(f"- {fid}: {msg}")

        if price_series:
            st.session_state["price_series"] = price_series
            st.success(f"Hämtade data för {len(price_series)} fonder.")
            st.write("Exempel på prisdata:")
            st.dataframe(
                pd.concat(price_series, axis=1).tail().style.format("{:.2f}"),
                use_container_width=True,
            )
        else:
            st.error("Ingen fonddata kunde hämtas.")

# ---- STEG 2: Välj & klassificera ----
price_series = st.session_state["price_series"]

if price_series is not None:
    st.subheader("Steg 2: Välj fonder och klassificera")

    all_names = list(price_series.keys())
    selected_names = st.multiselect(
        "Välj fonder som ska ingå i analysen",
        options=all_names,
        default=all_names,
    )

    if len(selected_names) >= 2:
        cols = st.columns(2)
        fund_types = st.session_state["fund_types"]

        st.markdown("Markera typ för varje fond:")

        fund_types = st.session_state["fund_types"]
        loaded_config = st.session_state.get("loaded_config")  # 👈 från file_uploader i sidebar

        st.markdown("Markera typ för varje fond:")

        for i, name in enumerate(selected_names):
            with cols[i % 2]:
                cfg_type = None

                # Försök hitta typ i laddad konfiguration (om sådan finns)
                if loaded_config is not None:
                    fund_id = None
                    if "(" in name and name.endswith(")"):
                        fund_id = name.split("(")[-1].strip(")")
                    if fund_id:
                        for item in loaded_config:
                            if str(item.get("fund_id")) == fund_id:
                                cfg_type = item.get("type")
                                break

                if cfg_type in ["Räntefond", "Svensk aktiefond", "Utländsk aktiefond"]:
                    default_type = cfg_type
                else:
                    # fallback: gissa som tidigare
                    default_type = fund_types.get(
                        name,
                        "Svensk aktiefond" if ("Sverige" in name or "OMX" in name) else
                        ("Utländsk aktiefond" if ("Global" in name or "World" in name or "USA" in name) else "Räntefond")
                    )

                choice = st.selectbox(
                    f"Typ för {name}",
                    ["Räntefond", "Svensk aktiefond", "Utländsk aktiefond"],
                    index=["Räntefond", "Svensk aktiefond", "Utländsk aktiefond"].index(default_type),
                )
                fund_types[name] = choice


        # Efter att fund_types fyllts på i steg 2
        st.subheader("Spara fondval & klassificering")

        if st.button("💾 Skapa konfigurationsfil"):
            config = []
            for name in selected_names:
                ftype = fund_types[name]
                # Plocka ut Avanza-ID ur namnet: "Fondnamn (12345)"
                fund_id = None
                if "(" in name and name.endswith(")"):
                    fund_id = name.split("(")[-1].strip(")")
                config.append(
                    {
                        "fund_id": fund_id,
                        "type": ftype,
                        "label": name,
                    }
                )

            json_str = json.dumps(config, ensure_ascii=False, indent=2)

            st.download_button(
                "Ladda ner fondkonfiguration",
                data=json_str,
                file_name="fondkonfiguration.json",
                mime="application/json",
            )

        # ---- STEG 3: Kör simulering ----
        st.subheader("Steg 3: Kör portföljsimulering")
        run_button = st.button("▶ Kör simulering")

        if run_button:
            selected_prices = {name: price_series[name] for name in selected_names}
            returns = build_return_matrix(selected_prices)

            st.write("Månadsavkastningar (sista 5 rader):")
            st.dataframe(
                (returns.tail() * 100).round(2).style.format("{:.2f}%"),
                use_container_width=True,
            )

            fund_types_now = st.session_state["fund_types"]
            swe_eq_mask = np.array(
                [fund_types_now[name] == "Svensk aktiefond" for name in selected_names]
            )
            for_eq_mask = np.array(
                [fund_types_now[name] == "Utländsk aktiefond" for name in selected_names]
            )
            n_swe = swe_eq_mask.sum()
            n_for = for_eq_mask.sum()
            n_eq = n_swe + n_for
            n_bond = len(selected_names) - n_eq

            min_bond_share = 1.0 - max_equity_share

            # feasibility checks
            if len(selected_names) * max_weight < 1.0 - 1e-9:
                st.error(
                    f"Max {max_weight*100:.1f} % per fond med endast {len(selected_names)} fonder "
                    f"kan inte summera till 100 %. Lägg till fler fonder eller höj max vikt."
                )
                st.stop()

            if n_bond == 0 and min_bond_share > 1e-9:
                st.error(
                    f"Inga räntefonder valda men max aktieandel är {max_equity_share*100:.1f} % "
                    f"(dvs minst {min_bond_share*100:.1f} % ränta krävs)."
                )
                st.stop()

            if n_eq > 0 and n_swe == 0 and max_foreign_equity_frac < 1.0 - 1e-9:
                st.error(
                    "Du har endast utländska aktiefonder men kräver att utländska aktier "
                    f"max får vara {max_foreign_equity_frac*100:.1f} % av aktiedelen. "
                    "Det är omöjligt – välj minst en svensk aktiefond eller höj gränsen."
                )
                st.stop()

            if n_bond > 0 and n_bond * max_weight < min_bond_share - 1e-9:
                st.error(
                    f"Du kräver minst {min_bond_share*100:.1f} % ränta (via max aktieandel "
                    f"{max_equity_share*100:.1f} %), men med {n_bond} räntefonder "
                    f"och max {max_weight*100:.1f} % per fond kan du som mest ha "
                    f"{n_bond*max_weight*100:.1f} % ränta."
                )
                st.stop()

            sim = simulate_portfolios(
                returns=returns,
                swe_eq_mask=swe_eq_mask,
                for_eq_mask=for_eq_mask,
                max_weight=max_weight,
                max_equity_share=max_equity_share,
                max_foreign_equity_frac=max_foreign_equity_frac,
                n_portfolios=n_portfolios,
                rf=rf,
            )

            port_returns = sim["returns"]
            port_vols = sim["vols"]
            sharpes = sim["sharpes"]
            weights = sim["weights"]
            mean_returns = sim["mean_returns"]
            cov_matrix = sim["cov_matrix"]
            asset_names = returns.columns.tolist()

            min_idx = np.argmin(port_vols)
            max_idx = np.nanargmax(sharpes)

            w_min = weights[min_idx]
            w_max = weights[max_idx]

            min_desc = describe_portfolio(w_min, asset_names, mean_returns, cov_matrix)
            max_desc = describe_portfolio(w_max, asset_names, mean_returns, cov_matrix)

            # plot
            st.subheader("Resultat: Risk–avkastningsdiagram")

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(port_vols, port_returns, s=5, alpha=0.3, label="Slumpade portföljer")
            ax.scatter(
                min_desc["Årsvolatilitet"],
                min_desc["Förväntad årsavkastning"],
                color="green",
                s=80,
                label="Minsta varians",
                zorder=5,
            )
            ax.scatter(
                max_desc["Årsvolatilitet"],
                max_desc["Förväntad årsavkastning"],
                color="purple",
                s=80,
                label="Max Sharpe",
                zorder=5,
            )
            ax.set_xlabel("Risk (årsvolatilitet)")
            ax.set_ylabel("Förväntad årsavkastning")
            ax.set_title("Portföljsimulering med aktie-/ränte- & svensk/utländsk-constraints")
            ax.grid(True)
            ax.legend()
            st.pyplot(fig, clear_figure=True)

            # ------ Historisk portföljutveckling + framtidsprojektion ------

            st.subheader("Historisk och förväntad portföljutveckling")

            # Historiska månadsavkastningar
            hist = returns[selected_names]

            # Välj en portfölj att rita – t.ex. Max Sharpe
            chosen_w = w_max

            # Viktad historisk utveckling
            hist_port_ret = (hist * chosen_w).sum(axis=1)

            # Gör om till index (start 100)
            hist_index = (1 + hist_port_ret).cumprod() * 100

            # ----- Framtida projektion -----
            years_forward = 10
            periods = years_forward * 12

            expected_annual_return = max_desc["Förväntad årsavkastning"]
            expected_monthly = (1 + expected_annual_return) ** (1 / 12) - 1

            future_index = [hist_index.iloc[-1]]
            for _ in range(periods):
                future_index.append(future_index[-1] * (1 + expected_monthly))

            future_dates = pd.date_range(hist_index.index[-1], periods=periods + 1, freq="M")

            future_series = pd.Series(future_index, index=future_dates)

            # ----- Plotta -----
            fig_hist, ax_hist = plt.subplots(figsize=(10, 5))

            ax_hist.plot(hist_index.index, hist_index.values, label="Historisk utveckling")
            ax_hist.plot(future_series.index, future_series.values, label="Förväntad utveckling (10 år)",
                         linestyle="--")

            ax_hist.set_title("Historisk & framtida portföljutveckling")
            ax_hist.set_ylabel("Index (start = 100)")
            ax_hist.grid(True)
            ax_hist.legend()

            st.pyplot(fig_hist, clear_figure=True)

            # tabeller
            st.subheader("Nyckelportföljer")


            def _format(desc: pd.Series) -> pd.DataFrame:
                dfp = desc.to_frame("Värde").copy()

                # Markera vilka rader som är vikter (innan vi filtrerar)
                mask_w = dfp.index.str.startswith("Vikt ")

                # Gör vikterna numeriska (0–1)
                dfp.loc[mask_w, "Värde"] = dfp.loc[mask_w, "Värde"].astype(float)

                # Dela upp i "info-rader" och "viktrader"
                df_info = dfp[~mask_w]
                df_weights = dfp[mask_w]

                # Filtrera bort vikter under min_display_weight
                df_weights = df_weights[df_weights["Värde"] >= min_display_weight]

                # Skala vikterna till procent
                df_weights["Värde"] = df_weights["Värde"] * 100

                # Slå ihop igen: info överst, vikter under
                dfp_out = pd.concat([df_info, df_weights])

                return dfp_out


            st.markdown("### Minsta varians-portfölj")
            st.dataframe(_format(min_desc).style.format("{:.2f}"), use_container_width=True)

            st.markdown("### Max Sharpe-portfölj")
            st.dataframe(_format(max_desc).style.format("{:.2f}"), use_container_width=True)

            # ------ Pajdiagram över fondtyper (viktad med portföljvikter) -------
            st.subheader("Fondfördelning mellan fondtyper (viktad portfölj)")

            type_weights = {
                "Räntefond": 0.0,
                "Svensk aktiefond": 0.0,
                "Utländsk aktiefond": 0.0,
            }

            # använd t.ex. Max Sharpe-portföljen
            for i, name in enumerate(selected_names):
                ftype = fund_types[name]  # "Räntefond", "Svensk aktiefond", "Utländsk aktiefond"
                w = float(w_max[i])
                if w < min_display_weight:
                    continue  # hoppa över småvikter
                if ftype in type_weights:
                    type_weights[ftype] += w

            # ta bort fondtyper med 0 vikt
            type_weights = {k: v for k, v in type_weights.items() if v > 0}

            if type_weights:
                fig_pie, ax_pie = plt.subplots(figsize=(6, 6))
                ax_pie.pie(
                    list(type_weights.values()),
                    labels=list(type_weights.keys()),
                    autopct='%1.1f%%'
                )
                ax_pie.set_title("Fördelning mellan fondtyper i vald portfölj (≥ min vikt)")
                st.pyplot(fig_pie, clear_figure=True)
            else:
                st.info("Inga fondtyper över vald minimivikt att visa i pajdiagrammet.")



    elif len(selected_names) > 0:
        st.info("Välj minst två fonder för att kunna simulera.")
else:
    st.info("👉 Börja med att ladda fonder i Steg 1.")

