import datetime as dt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from config_io import (
    build_fund_type_map,
    create_config,
    load_config,
    parse_fund_id_from_label,
)
from data_loader import build_return_matrix, fetch_fund_history
from optimization import (
    efficient_frontier_constrained,
    efficient_frontier_unconstrained,
    portfolio_stats,
)
from portfolio_engine import describe_portfolio, simulate_portfolios

CONFIG_PATH = Path(__file__).resolve().parent / "fondkonfiguration.json"
FUND_TYPE_OPTIONS = ["Räntefond", "Svensk aktiefond", "Utländsk aktiefond"]


def guess_fund_type(
    name: str,
    fund_id: str | None,
    existing_types: dict[str, str],
    fund_type_map_by_id: dict[str, str],
) -> str:
    """
    Försöker välja en rimlig default-klassificering för en fond.

    Parametrar
    ----------
    name: str
        Fondnamn inkl. ID ("Fond (1234)").
    fund_id: str | None
        Avanza-ID om det går att utläsa.
    existing_types: dict[str, str]
        Tidigare valda typer (lagrade i sessionen).
    fund_type_map_by_id: dict[str, str]
        Typ-data hämtad från uppladdad konfiguration.

    Retur
    -----
    str
        En av FUND_TYPE_OPTIONS.
    """

    if fund_id and fund_id in fund_type_map_by_id:
        return fund_type_map_by_id[fund_id]

    if name in existing_types:
        return existing_types[name]

    if "Sverige" in name or "OMX" in name:
        return "Svensk aktiefond"
    if "Global" in name or "World" in name or "USA" in name:
        return "Utländsk aktiefond"
    return "Räntefond"


def sync_managed_funds_from_series(price_series: dict[str, pd.Series]) -> None:
    """Uppdaterar sessionens lista med fonder baserat på inladdade serier."""

    managed = []
    fund_types = st.session_state.get("fund_types", {})
    for label in price_series:
        managed.append(
            {
                "label": label,
                "fund_id": parse_fund_id_from_label(label),
                "type": fund_types.get(label),
            }
        )
    st.session_state["managed_funds"] = managed


# ------------------------------
# Streamlit layout
# ------------------------------

st.set_page_config(page_title="Portföljsimulator (Avanza)", layout="wide")
st.title("📊 Portföljsimulator med Avanza-fonder")

st.markdown(
    """
Flöde:

1. **Ladda fonder** via Avanza-ID eller konfigurationsfil
2. **Välj & klassificera** (räntefond / svensk aktiefond / utländsk aktiefond)
3. **Kör analys** med:
   - Monte Carlo (med dina constraints)
   - Teoretisk effektiv front (utan constraints)
   - Effektiv front med constraints
   - Minsta varians- & Max Sharpe-portföljer (både teoretisk & constrained)
"""
)

# ------------------------------
# Session state
# ------------------------------

if "price_series" not in st.session_state:
    st.session_state["price_series"] = None
if "fund_types" not in st.session_state:
    st.session_state["fund_types"] = {}
if "loaded_config" not in st.session_state:
    st.session_state["loaded_config"] = None
if "fund_type_map_by_id" not in st.session_state:
    st.session_state["fund_type_map_by_id"] = {}
if "managed_funds" not in st.session_state:
    st.session_state["managed_funds"] = []

# ------------------------------
# SIDEBAR – generella inställningar
# ------------------------------

st.sidebar.header("Generella inställningar")

# Försök auto-ladda från fondkonfiguration.json i projektroten
if st.session_state["loaded_config"] is None and CONFIG_PATH.exists():
    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as f:
            loaded_config = load_config(f)
        st.session_state["loaded_config"] = loaded_config
        st.session_state["fund_type_map_by_id"] = build_fund_type_map(loaded_config)
        if not st.session_state["managed_funds"]:
            st.session_state["managed_funds"] = loaded_config
        st.sidebar.success(f"Laddade {CONFIG_PATH.name} från disk.")
    except Exception as e:
        st.sidebar.warning(f"Kunde inte läsa {CONFIG_PATH.name}: {e}")

# Möjlighet att ladda annan fil manuellt
config_file = st.sidebar.file_uploader("Ladda annan JSON-fil (valfritt)", type=["json"])

if config_file is not None:
    try:
        loaded_config = load_config(config_file)
        st.session_state["loaded_config"] = loaded_config
        st.session_state["fund_type_map_by_id"] = build_fund_type_map(loaded_config)
        st.session_state["managed_funds"] = loaded_config

        ids_from_config = sorted(
            {str(item["fund_id"]) for item in loaded_config if item.get("fund_id")}
        )
        st.sidebar.success(f"Laddade {len(ids_from_config)} fonder från uppladdad fil.")
        st.sidebar.write("Fonder i filen:", ", ".join(ids_from_config))
    except Exception as e:
        st.sidebar.error(f"Kunde inte läsa uppladdad konfiguration: {e}")


# Gemensamma parametrar

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
    st.sidebar.number_input("Antal Monte Carlo-portföljer", 1000, 50000, 5000, step=1000)
)

show_monte_carlo = st.sidebar.checkbox("Visa Monte Carlo-spridning", value=True)
show_theoretical = st.sidebar.checkbox("Visa teoretisk (unconstrained) front", value=True)


# ------------------------------
# TABS: Fondhantering & Analys
# ------------------------------

tab_manage, tab_analysis = st.tabs(["Fondhantering", "Analys"])

with tab_manage:
    st.subheader("Hantera fonder: lägg till, ta bort och klassificera")
    st.write(
        "Lägg till fond-ID:n, ladda sparade konfigurationer och uppdatera klassificeringar "
        "innan du kör analysen."
    )

    loaded_config = st.session_state.get("loaded_config")
    fund_type_map_by_id = st.session_state.get("fund_type_map_by_id", {})

    st.markdown("### Lägg till fonder")
    ids_input = st.text_area(
        "Avanza fond-ID (komma- eller radseparerade)",
        value="",
        key="ids_input_area",
    )

    include_loaded_config = st.checkbox(
        "Ta med fonder från laddad konfigurationsfil",
        value=loaded_config is not None,
    )

    load_button = st.button("🔄 Hämta & lägg till fonder")

    if load_button:
        ids_text = [x.strip() for x in ids_input.replace(",", "\n").splitlines() if x.strip()]

        ids_from_cfg = []
        if include_loaded_config and loaded_config is not None:
            ids_from_cfg = [
                str(item["fund_id"]) for item in loaded_config if item.get("fund_id")
            ]

        ids = sorted(set(ids_text + ids_from_cfg))

        if not ids:
            st.error(
                "Du måste ange minst ett Avanza-ID eller använda fonder från konfigurationen."
            )
        else:
            price_series = st.session_state.get("price_series") or {}
            errors = []

            from_str = start_date.isoformat()
            to_str = end_date.isoformat()

            progress = st.progress(0.0)
            for i, fid in enumerate(ids, start=1):
                try:
                    s = fetch_fund_history(fid, from_str, to_str)
                    price_series[s.name] = s

                    default_type = guess_fund_type(
                        s.name,
                        parse_fund_id_from_label(s.name),
                        st.session_state.get("fund_types", {}),
                        fund_type_map_by_id,
                    )
                    st.session_state["fund_types"][s.name] = default_type
                except Exception as e:
                    errors.append((fid, str(e)))
                progress.progress(i / len(ids))

            if errors:
                st.warning("Kunde inte läsa in vissa fonder:")
                for fid, msg in errors:
                    st.write(f"- {fid}: {msg}")

            if price_series:
                st.session_state["price_series"] = price_series
                sync_managed_funds_from_series(price_series)
                st.success(f"Hämtade data för {len(price_series)} fonder.")
            else:
                st.error("Ingen fonddata kunde hämtas.")

    price_series = st.session_state.get("price_series")

    if price_series:
        st.markdown("### Aktiva fonder & klassificering")
        managed_rows = []
        for label, serie in price_series.items():
            managed_rows.append(
                {
                    "Fond": label,
                    "Avanza-ID": parse_fund_id_from_label(label),
                    "Typ": st.session_state["fund_types"].get(label)
                    or guess_fund_type(
                        label,
                        parse_fund_id_from_label(label),
                        st.session_state.get("fund_types", {}),
                        fund_type_map_by_id,
                    ),
                }
            )

        managed_df = pd.DataFrame(managed_rows)
        edited_df = st.data_editor(
            managed_df,
            key="fund_editor",
            hide_index=True,
            column_config={
                "Typ": st.column_config.SelectboxColumn(
                    "Typ",
                    options=FUND_TYPE_OPTIONS,
                    help="Klassificering används i constraints vid analys.",
                )
            },
        )

        updated_types = {
            row["Fond"]: row["Typ"] for _, row in edited_df.iterrows() if row["Typ"]
        }
        st.session_state["fund_types"] = updated_types
        sync_managed_funds_from_series(price_series)

        st.markdown("### Ta bort fonder")
        to_remove = st.multiselect(
            "Markera fonder som ska tas bort från analysen",
            options=list(price_series.keys()),
        )
        if st.button("🗑 Ta bort valda fonder") and to_remove:
            for label in to_remove:
                st.session_state["fund_types"].pop(label, None)
                if price_series and label in price_series:
                    price_series.pop(label, None)
            st.session_state["price_series"] = price_series if price_series else None
            sync_managed_funds_from_series(price_series or {})
            st.success(f"Tog bort {len(to_remove)} fonder.")

        st.markdown("### Spara fondval & klassificering")
        if st.button("💾 Skapa / uppdatera konfigurationsfil"):
            labels = list(price_series.keys())
            json_str = create_config(labels, st.session_state["fund_types"])

            try:
                CONFIG_PATH.write_text(json_str, encoding="utf-8")
                st.success(f"Konfiguration sparad till {CONFIG_PATH.name} i projektroten.")
            except Exception as e:
                st.error(f"Kunde inte spara till {CONFIG_PATH.name}: {e}")

            st.download_button(
                "Ladda ner fondkonfiguration",
                data=json_str,
                file_name="fondkonfiguration.json",
                mime="application/json",
            )

        st.markdown("### Senaste prisdata (sista 5 värden)")
        st.dataframe(
            pd.concat(price_series, axis=1).tail().style.format("{:.2f}"),
            width="stretch",
        )
    else:
        st.info("Lägg till eller ladda fonder för att komma igång.")


with tab_analysis:
    price_series = st.session_state.get("price_series")

    if price_series:
        st.subheader("Välj fonder för analys")
        all_names = list(price_series.keys())
        selected_names = st.multiselect(
            "Fonder som ska ingå i simuleringen",
            options=all_names,
            default=all_names,
            key="analysis_funds",
        )

        if len(selected_names) >= 2:
            fund_types = st.session_state.get("fund_types", {})
            fund_type_map_by_id = st.session_state.get("fund_type_map_by_id", {})
            for name in selected_names:
                if name not in fund_types:
                    fund_types[name] = guess_fund_type(
                        name,
                        parse_fund_id_from_label(name),
                        fund_types,
                        fund_type_map_by_id,
                    )
            st.session_state["fund_types"] = fund_types

            st.subheader("Kör portföljsimulering & effektiv front")
            run_button = st.button("▶ Kör analys")

            if run_button:
                selected_prices = {name: price_series[name] for name in selected_names}
                returns = build_return_matrix(selected_prices, freq="ME")

                st.write("Månadsavkastningar (sista 5 rader):")
                st.dataframe(
                    (returns.tail() * 100).round(2).style.format("{:.2f}%"),
                    width="stretch",
                )

                mean_returns = returns.mean() * 12
                cov_matrix = returns.cov() * 12
                asset_names = returns.columns.tolist()

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
                weights_mc = sim["weights"]

                mc_min_idx = np.argmin(port_vols)
                mc_max_idx = np.nanargmax(sharpes)
                w_mc_min = weights_mc[mc_min_idx]
                w_mc_max = weights_mc[mc_max_idx]

                mc_min_desc = describe_portfolio(
                    w_mc_min, asset_names, mean_returns, cov_matrix
                )
                mc_max_desc = describe_portfolio(
                    w_mc_max, asset_names, mean_returns, cov_matrix
                )

                w_min_unc = w_max_unc = None
                front_risk_unc = front_ret_unc = np.array([])

                try:
                    front_risk_unc, front_ret_unc, w_min_unc, w_max_unc = (
                        efficient_frontier_unconstrained(mean_returns, cov_matrix, rf)
                    )
                except Exception as e:
                    st.warning(f"Kunde inte beräkna teoretisk (unconstrained) front: {e}")

                w_min_con = w_max_con = None
                front_risk_con = front_ret_con = np.array([])

                try:
                    front_risk_con, front_ret_con, w_min_con, w_max_con = (
                        efficient_frontier_constrained(
                            mean_returns,
                            cov_matrix,
                            rf,
                            swe_eq_mask=swe_eq_mask,
                            for_eq_mask=for_eq_mask,
                            max_weight=max_weight,
                            max_equity_share=max_equity_share,
                            max_foreign_equity_frac=max_foreign_equity_frac,
                        )
                    )
                except Exception as e:
                    st.warning(f"Kunde inte beräkna constrained front: {e}")

                min_desc_unc = max_desc_unc = None
                min_desc_con = max_desc_con = None

                if w_min_unc is not None:
                    r_min_unc, v_min_unc = portfolio_stats(
                        w_min_unc, mean_returns, cov_matrix
                    )
                    min_desc_unc = describe_portfolio(
                        w_min_unc, asset_names, mean_returns, cov_matrix
                    )
                else:
                    r_min_unc = v_min_unc = np.nan

                if w_max_unc is not None:
                    r_max_unc, v_max_unc = portfolio_stats(
                        w_max_unc, mean_returns, cov_matrix
                    )
                    max_desc_unc = describe_portfolio(
                        w_max_unc, asset_names, mean_returns, cov_matrix
                    )
                else:
                    r_max_unc = v_max_unc = np.nan

                if w_min_con is not None:
                    r_min_con, v_min_con = portfolio_stats(
                        w_min_con, mean_returns, cov_matrix
                    )
                    min_desc_con = describe_portfolio(
                        w_min_con, asset_names, mean_returns, cov_matrix
                    )
                else:
                    r_min_con = v_min_con = np.nan

                if w_max_con is not None:
                    r_max_con, v_max_con = portfolio_stats(
                        w_max_con, mean_returns, cov_matrix
                    )
                    max_desc_con = describe_portfolio(
                        w_max_con, asset_names, mean_returns, cov_matrix
                    )
                else:
                    r_max_con = v_max_con = np.nan

                st.subheader("Resultat: Risk–avkastningsdiagram")

                fig, ax = plt.subplots(figsize=(8, 6))

                if show_monte_carlo:
                    ax.scatter(
                        port_vols,
                        port_returns,
                        s=3,
                        alpha=0.25,
                        label="Monte Carlo-portföljer (constrained)",
                    )

                if front_risk_con.size > 0:
                    ax.plot(
                        front_risk_con,
                        front_ret_con,
                        color="tab:orange",
                        linewidth=2,
                        label="Effektiv front (constrained)",
                    )

                if show_theoretical and front_risk_unc.size > 0:
                    ax.plot(
                        front_risk_unc,
                        front_ret_unc,
                        color="tab:blue",
                        linewidth=2,
                        label="Effektiv front (teoretisk, unconstrained)",
                    )

                if show_theoretical and w_min_unc is not None:
                    ax.scatter(
                        v_min_unc,
                        r_min_unc,
                        color="blue",
                        marker="o",
                        s=80,
                        label="Minsta varians (teori)",
                    )
                if show_theoretical and w_max_unc is not None:
                    ax.scatter(
                        v_max_unc,
                        r_max_unc,
                        color="blue",
                        marker="x",
                        s=80,
                        label="Max Sharpe (teori)",
                    )

                if w_min_con is not None:
                    ax.scatter(
                        v_min_con,
                        r_min_con,
                        color="orange",
                        marker="o",
                        s=80,
                        label="Minsta varians (constrained)",
                    )
                if w_max_con is not None:
                    ax.scatter(
                        v_max_con,
                        r_max_con,
                        color="orange",
                        marker="x",
                        s=80,
                        label="Max Sharpe (constrained)",
                    )

                ax.set_xlabel("Risk (årsvolatilitet)")
                ax.set_ylabel("Förväntad årsavkastning")
                ax.set_title(
                    "Portföljer: Monte Carlo & effektiv front\n"
                    "(teoretisk vs constrained med dina regler)"
                )
                ax.grid(True)
                ax.legend()
                st.pyplot(fig, clear_figure=True)

                st.subheader("Förväntad utveckling över 10 år")

                years = np.arange(0, 11)
                fig_hist, ax_hist = plt.subplots(figsize=(8, 4))
                init_invest = 100

                def plot_growth(weights: np.ndarray, label: str, color: str) -> None:
                    r, v = portfolio_stats(weights, mean_returns, cov_matrix)
                    # antag lognormal approx för enkel visualisering
                    future = init_invest * np.exp(
                        (r - (v**2) / 2) * years + v * np.sqrt(years)
                    )
                    ax_hist.plot(years, future, label=label, color=color)

                if w_max_unc is not None and show_theoretical:
                    plot_growth(w_max_unc, "Teoretisk Max Sharpe", "blue")
                if w_max_con is not None:
                    plot_growth(w_max_con, "Max Sharpe (constrained)", "orange")

                if show_monte_carlo:
                    future_series = init_invest * (
                        1 + pd.Series(port_returns).sort_values().rolling(50).mean().fillna(0)
                    )
                    ax_hist.plot(
                        future_series.index,
                        future_series.values,
                        label="Förväntad utveckling (10 år)",
                        linestyle="--",
                    )
                ax_hist.set_title(
                    "Historisk & framtida portföljutveckling – Max Sharpe-portfölj (constrained)"
                )
                ax_hist.set_ylabel("Index (start = 100)")
                ax_hist.grid(True)
                ax_hist.legend()
                st.pyplot(fig_hist, clear_figure=True)

                st.subheader("Nyckelportföljer (teori, constrained & Monte Carlo)")

                def _format(desc: pd.Series) -> pd.DataFrame:
                    """Konverterar describe_portfolio-Serie till tabell, filtrerar småvikter."""

                    dfp = desc.to_frame("Värde").copy()
                    mask_w = dfp.index.str.startswith("Vikt ")

                    df_info = dfp[~mask_w]
                    df_weights = dfp[mask_w].copy()

                    if not df_weights.empty:
                        df_weights["Värde"] = df_weights["Värde"].astype(float)
                        df_weights = df_weights[df_weights["Värde"] >= min_display_weight]
                        df_weights["Värde"] = df_weights["Värde"] * 100

                    return pd.concat([df_info, df_weights])

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("### Teoretisk (unconstrained)")
                    if min_desc_unc is not None:
                        st.markdown("**Minsta varians**")
                        st.dataframe(
                            _format(min_desc_unc).style.format("{:.2f}"), width="stretch"
                        )
                    if max_desc_unc is not None:
                        st.markdown("**Max Sharpe**")
                        st.dataframe(
                            _format(max_desc_unc).style.format("{:.2f}"), width="stretch"
                        )

                with col2:
                    st.markdown("### Faktisk (constrained)")
                    if min_desc_con is not None:
                        st.markdown("**Minsta varians**")
                        st.dataframe(
                            _format(min_desc_con).style.format("{:.2f}"), width="stretch"
                        )
                    if max_desc_con is not None:
                        st.markdown("**Max Sharpe**")
                        st.dataframe(
                            _format(max_desc_con).style.format("{:.2f}"), width="stretch"
                        )

                st.markdown("### Monte Carlo – bästa portföljer (simulerade)")
                st.markdown("**Minsta varians (lägst risk bland simulerade)**")
                st.dataframe(_format(mc_min_desc).style.format("{:.2f}"), width="stretch")

                st.markdown("**Max Sharpe (bäst risk/avkastning bland simulerade)**")
                st.dataframe(_format(mc_max_desc).style.format("{:.2f}"), width="stretch")

                st.subheader("Fondfördelning mellan fondtyper (Max Sharpe)")

                def type_weight_dict(weights: np.ndarray) -> dict[str, float]:
                    tw = {
                        "Räntefond": 0.0,
                        "Svensk aktiefond": 0.0,
                        "Utländsk aktiefond": 0.0,
                    }
                    for i, name in enumerate(selected_names):
                        ftype = fund_types[name]
                        w = float(weights[i])
                        if w < min_display_weight:
                            continue
                        if ftype in tw:
                            tw[ftype] += w
                    return {k: v for k, v in tw.items() if v > 0}

                col_p1, col_p2 = st.columns(2)

                with col_p1:
                    st.markdown("**Teoretisk Max Sharpe (unconstrained)**")
                    if w_max_unc is not None:
                        tw_unc = type_weight_dict(w_max_unc)
                        if tw_unc:
                            fig_pie1, ax_pie1 = plt.subplots(figsize=(5, 5))
                            ax_pie1.pie(
                                list(tw_unc.values()),
                                labels=list(tw_unc.keys()),
                                autopct="%1.1f%%",
                            )
                            ax_pie1.set_title("Fondtyper – teoretisk")
                            st.pyplot(fig_pie1, clear_figure=True)
                        else:
                            st.info("Inga vikter över min viktnivå för teoretisk portfölj.")
                    else:
                        st.info("Teoretisk Max Sharpe kunde inte beräknas.")

                with col_p2:
                    st.markdown("**Max Sharpe (constrained)**")
                    if w_max_con is not None:
                        tw_con = type_weight_dict(w_max_con)
                        if tw_con:
                            fig_pie2, ax_pie2 = plt.subplots(figsize=(5, 5))
                            ax_pie2.pie(
                                list(tw_con.values()),
                                labels=list(tw_con.keys()),
                                autopct="%1.1f%%",
                            )
                            ax_pie2.set_title("Fondtyper – constrained")
                            st.pyplot(fig_pie2, clear_figure=True)
                        else:
                            st.info("Inga vikter över min viktnivå för constrained portfölj.")
                    else:
                        st.info("Constrained Max Sharpe kunde inte beräknas.")

        elif len(selected_names) > 0:
            st.info("Välj minst två fonder för att kunna simulera.")
    else:
        st.info("👉 Lägg till fonder i fliken 'Fondhantering' för att kunna analysera.")
