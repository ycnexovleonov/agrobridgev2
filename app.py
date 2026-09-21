import json
from datetime import datetime, timedelta

import folium
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from folium.raster_layers import ImageOverlay
from streamlit_folium import st_folium

from backtest import run_backtest
from data_cache import load_snapshot, load_snapshot_raster, snapshot_available
from ai_analysis import CHECKLIST_ITEMS, assess_risk, classify_anomaly, forecast_ndvi, primary_action
from analysis import (baseline_at, compare_seasons, compute_seasonal_norm,
                      detect_anomalies, shift_year, summarize)
from config import (DEFAULT_FIELD_GEOJSON, HISTORY_YEARS, NON_CROP_NDVI,
                    SEASON_END, SEASON_START)
from export import export_csv, export_geojson, export_pdf_report
from i18n import t
from mock_data import generate_mock_timeseries
from raster import (NDVI_COLOR_MAX, NDVI_COLOR_MIN, extract_problem_zones,
                    field_area_ha, folium_bounds, legend_colors,
                    mock_ndvi_raster, ndvi_raster_to_png, normalize_field_geometry,
                    png_to_data_uri, problem_zone_mask_png, zone_statistics)

st.set_page_config(page_title="Agrobridge", page_icon="🌾", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {font-family: 'Manrope', sans-serif;}

    .block-container {padding-top: 4rem; max-width: 1400px;}
    h1, h2, h3 {letter-spacing: -0.01em;}

    [data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #DCE1D6;
        border-left: 3px solid #2F6B3A;
        border-radius: 10px;
        padding: 0.7rem 0.9rem;
    }
    [data-testid="stMetricLabel"] p {font-size: 0.78rem; color: #55605A;}
    .stTabs [data-baseweb="tab"] {padding: 0.5rem 1.2rem; font-weight: 600;}
    .stTabs [data-baseweb="tab-list"] {gap: 4px;}
    div[data-testid="stSidebarUserContent"] {padding-top: 0.6rem;}
    [data-testid="stSidebar"] {border-right: 1px solid #E4E8E0;}

    /* Фирменный хедер вместо голого st.title */
    .agro-hero {
        background: linear-gradient(120deg, #2E7D32 0%, #43964A 55%, #6BAF5A 100%);
        border-radius: 18px;
        padding: 1.3rem 1.6rem;
        margin-bottom: 1.1rem;
        color: #FFFFFF;
        box-shadow: 0 4px 16px rgba(46,125,50,0.22);
    }
    .agro-hero-title {
        font-size: 1.65rem; font-weight: 800; display: flex;
        align-items: center; gap: 0.5rem; margin-bottom: 0.15rem;
    }
    .agro-hero-sub {font-size: 0.92rem; opacity: 0.92; font-weight: 500;}
    .agro-hero-badges {margin-top: 0.65rem; display: flex; gap: 0.5rem; flex-wrap: wrap;}
    .agro-hero-badge {
        background: rgba(255,255,255,0.18); border: 1px solid rgba(255,255,255,0.35);
        border-radius: 999px; padding: 0.22rem 0.8rem; font-size: 0.8rem; font-weight: 600;
    }

    /* Заголовки секций сайдбара */
    .agro-sidebar-section {
        font-size: 0.72rem; font-weight: 800; text-transform: uppercase;
        letter-spacing: 0.06em; color: #4F7A54; margin: 1rem 0 0.3rem;
    }

    .agro-card {
        background: #FFFFFF;
        border-radius: 14px;
        padding: 0.9rem 1.1rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.07);
        border-top: 4px solid var(--agro-accent, #2E7D32);
        height: 100%;
    }
    .agro-card .agro-card-label {
        font-size: 0.76rem; color: #5A655C; text-transform: uppercase;
        letter-spacing: 0.02em; margin-bottom: 0.15rem;
    }
    .agro-card .agro-card-value {
        font-size: 1.55rem; font-weight: 700; color: #1A1F1B; line-height: 1.15;
    }
    .agro-card .agro-card-sub {font-size: 0.78rem; color: #6B756D; margin-top: 0.2rem;}
    .agro-chip {
        display: inline-block; padding: 0.2rem 0.75rem; border-radius: 999px;
        font-weight: 700; font-size: 0.85rem; color: #FFFFFF;
    }
    .agro-checklist-item {
        padding: 0.4rem 0.7rem; border-radius: 10px; margin-bottom: 0.35rem;
        background: #F4F6F2; font-size: 0.88rem;
    }
    .agro-checklist-item.primary {
        background: #FDEEDB; border-left: 3px solid #F57C00; font-weight: 600;
    }

    /* Карточка-фича на вкладке "О проекте" */
    .agro-feature {
        background: #F7F9F5; border-radius: 12px; padding: 0.8rem 1rem;
        margin-bottom: 0.6rem; border-left: 3px solid #6BAF5A;
    }
    .agro-feature b {color: #2E7D32;}
    .agro-tech-chip {
        display: inline-block; background: #EEF2EA; color: #3A4A3C;
        border-radius: 8px; padding: 0.25rem 0.65rem; margin: 0.15rem;
        font-size: 0.82rem; font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

AGRO_GREEN = "#2E7D32"
AGRO_ORANGE = "#F57C00"
AGRO_RED = "#C62828"
RISK_COLORS = {"high": AGRO_RED, "medium": AGRO_ORANGE, "low": AGRO_GREEN, "unknown": "#9AA39B"}


def agro_card(label: str, value: str, sub: str = "", accent: str = AGRO_GREEN):
    """Скруглённая карточка метрики в фирменных цветах."""
    st.markdown(f"""
    <div class="agro-card" style="--agro-accent: {accent};">
        <div class="agro-card-label">{label}</div>
        <div class="agro-card-value">{value}</div>
        <div class="agro-card-sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)


def agro_chip(text: str, color: str) -> str:
    return f'<span class="agro-chip" style="background:{color};">{text}</span>'


def sidebar_section(title: str):
    st.markdown(f'<div class="agro-sidebar-section">{title}</div>', unsafe_allow_html=True)


def render_ndvi_legend(lang: str):
    """
    Компактная горизонтальная шкала NDVI — отдельным блоком ПОД картой,
    а не поверх неё (раньше это была плашка branca.colormap, накладываемая
    на сам холст Folium и перекрывающая часть растра).

    Цвета берутся из той же палитры legend_colors(), что красит сам растр
    (ndvi_raster_to_png) — значения на шкале гарантированно совпадают
    с цветами на карте, а не рисуются заново на глаз.
    """
    n = 7
    colors = legend_colors(n)
    step = (NDVI_COLOR_MAX - NDVI_COLOR_MIN) / (n - 1)
    ticks = [round(NDVI_COLOR_MIN + i * step, 2) for i in range(n)]
    gradient = ", ".join(colors)

    ticks_html = "".join(f'<span>{v:.2f}</span>' for v in ticks)

    st.markdown(f"""
    <div class="agro-card" style="--agro-accent: #8A938C; max-width: 560px;
                margin: 0.6rem auto 0; padding: 0.6rem 1rem;">
        <div style="display:flex; justify-content:space-between;
                    font-size:0.74rem; color:#444; padding:0 2px; margin-bottom:3px;">
            {ticks_html}
        </div>
        <div style="height:12px; border-radius:6px;
                    background: linear-gradient(to right, {gradient});"></div>
        <div style="text-align:center; font-size:0.78rem; color:#5A655C; margin-top:5px;">
            {t('legend_caption', lang)}
        </div>
    </div>
    """, unsafe_allow_html=True)


if "lang" not in st.session_state:
    st.session_state["lang"] = "ru"
lang = st.session_state["lang"]

st.markdown(f"""
<div class="agro-hero">
    <div class="agro-hero-title">🌾 {t("app_title", lang).replace("🌾 ", "")}</div>
    <div class="agro-hero-sub">{t("hero_subtitle", lang)}</div>
    <div class="agro-hero-badges">
        <span class="agro-hero-badge">Sentinel-2 · Copernicus</span>
        <span class="agro-hero-badge">NDVI / NDMI</span>
        <span class="agro-hero-badge">GeoJSON · CSV · PDF</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------- Сайдбар ----------
with st.sidebar:
    st.markdown("### 🌾 Agrobridge")
    lang_choice = st.radio(t("language", lang), ["Русский", "Қазақша"],
                           index=0 if lang == "ru" else 1, horizontal=True)
    new_lang = "ru" if lang_choice == "Русский" else "kk"
    if new_lang != lang:
        st.session_state["lang"] = new_lang
        st.rerun()

    sidebar_section(t("sidebar_field_section", lang))
    field_name = st.text_input(t("field_name", lang), value="Поле №1 (Акмолинская обл.)")
    uploaded = st.file_uploader(t("upload_geojson", lang), type=["geojson", "json"])

    sidebar_section(t("sidebar_data_section", lang))
    snapshot_ok = snapshot_available()
    source_keys = (["snapshot"] if snapshot_ok else []) + ["demo", "live"]
    source_labels = {"snapshot": t("data_source_snapshot", lang),
                     "demo": t("data_source_demo", lang),
                     "live": t("data_source_real", lang)}
    source = st.radio(t("data_source", lang), source_keys,
                      format_func=lambda k: source_labels[k],
                      help=t("data_source_help", lang))
    is_demo = source == "demo"
    if source == "snapshot":
        st.info(t("snapshot_info", lang))

    date_from = st.text_input(t("season_start", lang), value=SEASON_START)
    date_to = st.text_input(t("season_end", lang), value=SEASON_END)

    sidebar_section(t("sidebar_analysis_section", lang))
    use_history = st.checkbox(t("use_history", lang), value=True,
                              help=t("use_history_help", lang))
    compare_prev = st.checkbox(t("compare_prev_season", lang), value=False)
    show_raster = st.checkbox(t("show_raster", lang), value=True,
                              help=t("show_raster_help", lang))

    st.divider()
    run = st.button(t("calculate_btn", lang), type="primary", use_container_width=True)

# ---------- Контур поля ----------
geojson_error = None
if uploaded is not None:
    try:
        # getvalue(), а не json.load(uploaded): на повторных запусках скрипта
        # указатель файла может стоять в конце
        field_geojson = normalize_field_geometry(json.loads(uploaded.getvalue()))
    except (ValueError, KeyError, TypeError) as e:
        geojson_error = str(e)
        field_geojson = DEFAULT_FIELD_GEOJSON
else:
    field_geojson = DEFAULT_FIELD_GEOJSON


def parse_dates(d_from: str, d_to: str):
    """Возвращает (начало, конец) или None, если формат неверный или начало не раньше конца."""
    try:
        start = datetime.strptime(d_from.strip(), "%Y-%m-%d")
        end = datetime.strptime(d_to.strip(), "%Y-%m-%d")
    except ValueError:
        return None
    return (start, end) if start < end else None


def load_timeseries(d_from, d_to, demo, geojson):
    """
    Ряд текущего сезона. Возвращает (данные, режим, причина).

    Режим: "demo" — выбран демо; "real" — данные получены из Sentinel Hub;
    "fallback" — реальные данные запрошены, но получить их не удалось, и
    показаны демо-данные. Подмену нельзя делать молча, поэтому режим всегда
    выводится в интерфейсе.
    """
    if demo:
        return generate_mock_timeseries(d_from, d_to), "demo", None
    try:
        from sentinel_client import fetch_ndvi_ndmi_timeseries
        data = fetch_ndvi_ndmi_timeseries(geojson, d_from, d_to)
    except Exception as e:
        return generate_mock_timeseries(d_from, d_to), "fallback", f"{type(e).__name__}: {e}"
    if not data:
        return generate_mock_timeseries(d_from, d_to), "fallback", t("no_data_warning", lang)
    return data, "real", None


@st.cache_data(show_spinner=False)
def fetch_season(geojson_str: str, d_from: str, d_to: str):
    """
    Один прошлый сезон из Sentinel Hub. Кешируется только успешный ответ:
    исключение Streamlit не запоминает, поэтому после исправления ключей или
    сети повторный расчёт делает новый запрос, а не отдаёт закешированную ошибку.
    """
    from sentinel_client import fetch_ndvi_ndmi_timeseries
    return fetch_ndvi_ndmi_timeseries(json.loads(geojson_str), d_from, d_to)


def load_history(geojson_str: str, d_from: str, d_to: str, demo: bool, years: int):
    """Тот же период за прошлые сезоны. Возвращает (список сезонов, список ошибок)."""
    seasons, errors = [], []
    for k in range(1, years + 1):
        past_from, past_to = shift_year(d_from, -k), shift_year(d_to, -k)
        if demo:
            seasons.append(generate_mock_timeseries(past_from, past_to,
                                                    seed=100 + k, with_anomaly=False))
            continue
        try:
            data = fetch_season(geojson_str, past_from, past_to)
        except Exception as e:
            errors.append(f"{past_from[:4]}: {type(e).__name__}: {str(e)[:100]}")
            continue
        if data:
            seasons.append(data)
    return seasons, errors


@st.cache_data(show_spinner=False)
def load_raster(geojson_str: str, snapshot_date: str, demo: bool,
                demo_base: float = 0.6, demo_strength: float = 0.3, window_days: int = 5):
    """
    Растр NDVI на дату снимка. Результат кешируется: st_folium перезапускает
    скрипт при каждом взаимодействии с картой, и без кеша это был бы новый
    запрос к Process API на каждый клик. GeoJSON передаётся строкой, потому
    что аргументы кеша Streamlit должны быть хешируемыми.
    """
    geojson = json.loads(geojson_str)
    if demo:
        return mock_ndvi_raster(geojson, base=demo_base, strength=demo_strength)

    from raster import fetch_ndvi_raster
    center = datetime.strptime(snapshot_date, "%Y-%m-%d")
    d_from = (center - timedelta(days=window_days)).strftime("%Y-%m-%d")
    d_to = (center + timedelta(days=window_days)).strftime("%Y-%m-%d")
    return fetch_ndvi_raster(geojson, d_from, d_to)


def render_backtest(enriched_series, history_seasons, real_data: bool, lang: str):
    """
    Независимая проверка прогноза: скользящий backtest по всем сезонам и
    сравнение с наивными базовыми моделями (см. backtest.py).
    """
    st.subheader(t("bt_title", lang))
    result = run_backtest([enriched_series] + list(history_seasons), horizon_days=14)
    if "error" in result:
        st.info(t(f"bt_{result['error']}", lang))
        return

    st.caption(t("bt_intro", lang, h=result["horizon_days"]))
    if not real_data:
        st.warning(t("bt_demo_warn", lang))

    models = result["models"]
    label = lambda name: t(f"bt_model_{name}", lang)
    st.dataframe(pd.DataFrame([{
        t("bt_col_model", lang): label(name),
        "MAE": round(m["mae"], 4),
        "RMSE": round(m["rmse"], 4),
        t("bt_col_bias", lang): round(m["bias"], 4),
        t("bt_col_skill", lang): m["skill_vs_persistence_pct"],
    } for name, m in models.items()]), hide_index=True, use_container_width=True)

    fig = go.Figure(go.Bar(
        x=[label(n) for n in models], y=[m["mae"] for m in models.values()],
        marker_color=[AGRO_ORANGE if n == "linear" else "#8A938C" for n in models],
        text=[f"{m['mae']:.3f}" for m in models.values()], textposition="outside"))
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10),
                      plot_bgcolor="#FBFCFA", yaxis_title="MAE", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    if not result["has_norm"]:
        st.info(t("bt_no_norm", lang))
    if models["linear"]["mae"] < models["persistence"]["mae"]:
        st.success(t("bt_linear_better", lang, pct=models["linear"]["skill_vs_persistence_pct"]))
    else:
        st.warning(t("bt_linear_worse", lang))
    st.caption(t("bt_note", lang, n=result["n_samples"], s=result["n_seasons"]) + " "
               + t("bt_best_line", lang, best=label(result["best_model"])))


# ---------- Получение данных ----------
if "timeseries" not in st.session_state or run:
    snapshot = None
    if source == "snapshot":
        snapshot = load_snapshot()
        if snapshot is None:
            st.error(t("snapshot_broken", lang))
            st.stop()
        # контур и границы сезона берутся из снимка: растры и история
        # сохранены именно для них
        field_geojson = snapshot["meta"]["field_geojson"]
        date_from = snapshot["meta"]["date_from"]
        date_to = snapshot["meta"]["date_to"]

    if not parse_dates(date_from, date_to):
        st.error(t("bad_dates", lang))
        st.stop()

    with st.spinner(t("computing", lang)):
        if snapshot:
            raw_timeseries, data_mode, data_error = snapshot["timeseries"], "snapshot", None
        else:
            raw_timeseries, data_mode, data_error = load_timeseries(
                date_from, date_to, is_demo, field_geojson)

        # Если реальные данные недоступны, история тоже демонстрационная:
        # иначе три сезона подряд падали бы с той же ошибкой
        demo_history = data_mode not in ("real", "snapshot")

        history, history_errors = [], []
        if use_history:
            if snapshot:
                history = snapshot["history"]
            else:
                history, history_errors = load_history(json.dumps(field_geojson),
                                                       date_from, date_to, demo_history, HISTORY_YEARS)

        enriched, norm = detect_anomalies(raw_timeseries, history=history)
        ndmi_norm = compute_seasonal_norm(raw_timeseries, "ndmi_mean")
        summary = summarize(enriched)

        prev_timeseries = None
        if compare_prev:
            if history:
                prev_timeseries = history[0]
            elif data_mode == "snapshot":
                prev_timeseries = None
            elif demo_history:
                prev_timeseries = generate_mock_timeseries(
                    shift_year(date_from, -1), shift_year(date_to, -1),
                    seed=123, with_anomaly=False)
            else:
                prev_timeseries = load_timeseries(shift_year(date_from, -1),
                                                  shift_year(date_to, -1), False, field_geojson)[0]

        st.session_state.update({
            "timeseries": enriched, "norm": norm, "ndmi_norm": ndmi_norm,
            "summary": summary, "field_geojson": field_geojson,
            "prev_timeseries": prev_timeseries,
            "data_mode": data_mode, "data_error": data_error,
            "history_seasons": len(history), "history_errors": history_errors,
            "history_requested": use_history, "history": history,
            "snapshot_created": snapshot["meta"]["created_utc"] if snapshot else "",
            "snapshot_raster_dates": snapshot["meta"].get("raster_dates", []) if snapshot else [],
        })

enriched = st.session_state["timeseries"]
norm = st.session_state["norm"]
ndmi_norm = st.session_state["ndmi_norm"]
summary = st.session_state["summary"]
field_geojson = st.session_state["field_geojson"]
prev_timeseries = st.session_state.get("prev_timeseries")
data_mode = st.session_state.get("data_mode", "demo")
raster_is_demo = data_mode in ("demo", "fallback")
is_real_data = data_mode in ("real", "snapshot")

area_ha = field_area_ha(field_geojson)

if geojson_error:
    st.warning(t("geojson_error", lang, error=geojson_error))

if data_mode == "fallback":
    st.error(t("mode_fallback", lang, error=st.session_state.get("data_error") or "—"))
else:
    st.markdown(agro_chip(t(f"mode_{data_mode}", lang,
                            date=st.session_state.get("snapshot_created", "")),
                          AGRO_GREEN if is_real_data else AGRO_ORANGE),
                unsafe_allow_html=True)

mode_history = summary.get("norm_mode") == "history"
tab_monitoring, tab_zones, tab_ai, tab_reports = st.tabs([
    t("tab_monitoring", lang), t("tab_zones", lang), t("tab_ai", lang),
    t("tab_reports", lang),
])
# ================== МОНИТОРИНГ ==================
with tab_monitoring:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        agro_card(t("metric_area", lang), f"{area_ha} {t('ha', lang)}", accent=AGRO_GREEN)
    with col2:
        agro_card(t("metric_norm_hist" if mode_history else "metric_norm", lang),
                  str(norm), accent=AGRO_GREEN)
    with col3:
        sub3 = (f"{summary['low_quality_periods']} {t('no_data_periods', lang)}"
                if summary.get("low_quality_periods") else "")
        agro_card(t("metric_periods", lang), str(summary["total_periods"]), sub3,
                  accent="#9AA39B" if sub3 else AGRO_GREEN)
    with col4:
        sub4 = f"{summary['anomaly_periods']} {t('of_periods', lang)}"
        anomaly_accent = AGRO_RED if summary["anomaly_share_pct"] > 20 else (
            AGRO_ORANGE if summary["anomaly_share_pct"] > 0 else AGRO_GREEN)
        agro_card(t("metric_anomaly_share", lang), f"{summary['anomaly_share_pct']}%", sub4,
                  accent=anomaly_accent)

    if mode_history:
        st.caption(t("norm_history", lang, n=st.session_state.get("history_seasons", 0)))
    else:
        st.caption(t("norm_season", lang))
        if st.session_state.get("history_requested"):
            st.warning(t("history_missing", lang))
    if st.session_state.get("history_errors"):
        st.warning(t("history_error", lang, error="; ".join(st.session_state["history_errors"])))

    st.divider()
    left, right = st.columns([1, 1.3])

    with left:
        st.subheader(t("map_title", lang))

        by_date = {p["date"]: p for p in enriched}
        reliable = [p for p in enriched if not p.get("low_quality")]
        raster_date = None

        raster_candidates = reliable
        if data_mode == "snapshot":
            saved = set(st.session_state.get("snapshot_raster_dates", []))
            raster_candidates = [p for p in reliable if p["date"] in saved]

        if show_raster and raster_candidates:
            options = [p["date"] for p in raster_candidates]
            # по умолчанию — пик вегетации: неоднородность внутри поля на нём
            # информативнее всего
            default_date = max(raster_candidates, key=lambda p: p["ndvi_mean"])["date"]

            def option_label(d):
                p = by_date[d]
                if p.get("data_share_pct") is None:
                    return t("raster_option_noshare", lang, date=d, ndvi=p["ndvi_mean"])
                return t("raster_option", lang, date=d, ndvi=p["ndvi_mean"],
                         share=p["data_share_pct"])

            raster_date = st.selectbox(t("raster_date", lang), options,
                                       index=options.index(default_date),
                                       format_func=option_label,
                                       help=t("raster_date_help", lang))

        coords = field_geojson["coordinates"][0]
        center_lat = sum(c[1] for c in coords) / len(coords)
        center_lon = sum(c[0] for c in coords) / len(coords)

        last_point = reliable[-1] if reliable else (enriched[-1] if enriched else None)
        is_bad = last_point and last_point.get("is_anomaly")

        m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles=None)
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/"
                  "World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri World Imagery", name=t("layer_satellite", lang),
        ).add_to(m)
        folium.TileLayer("CartoDB positron", name=t("layer_scheme", lang)).add_to(m)

        st.session_state["zone_stats"] = None
        st.session_state["zones"] = []

        raster_rendered = False

        if raster_date:
            try:
                point = by_date[raster_date]
                if data_mode == "snapshot":
                    loaded = load_snapshot_raster(raster_date)
                    if loaded is None:
                        raise RuntimeError(t("snapshot_no_raster", lang, date=raster_date))
                    ndvi_arr, mask_arr = loaded
                else:
                    ndvi_arr, mask_arr = load_raster(
                        json.dumps(field_geojson), raster_date, raster_is_demo,
                        demo_base=float(point["ndvi_mean"]),
                        demo_strength=(0.45 if point.get("is_anomaly") else 0.15) * float(point["ndvi_mean"]),
                    )
                zstats = zone_statistics(ndvi_arr, mask_arr, field_area=area_ha)
                zstats["date"] = raster_date
                zones = (extract_problem_zones(ndvi_arr, mask_arr, zstats["threshold_ndvi"],
                                               field_geojson, area_ha)
                         if "threshold_ndvi" in zstats else [])
                st.session_state["zone_stats"] = zstats
                st.session_state["zones"] = zones

                bounds = folium_bounds(field_geojson)
                ImageOverlay(
                    image=png_to_data_uri(ndvi_raster_to_png(ndvi_arr, mask_arr)),
                    bounds=bounds, name=t("layer_ndvi", lang), opacity=1.0,
                ).add_to(m)

                if "threshold_ndvi" in zstats:
                    ImageOverlay(
                        image=png_to_data_uri(problem_zone_mask_png(
                            ndvi_arr, mask_arr, zstats["threshold_ndvi"])),
                        bounds=bounds, name=t("layer_zones", lang), opacity=1.0, show=False,
                    ).add_to(m)

                if zones:
                    outlines = folium.FeatureGroup(name=t("layer_zone_outlines", lang), show=True)
                    for zone in zones:
                        zp = zone["properties"]
                        folium.GeoJson(
                            zone,
                            style_function=lambda feature: {
                                "color": "#B4451F", "weight": 2.5,
                                "fillColor": "#B4451F", "fillOpacity": 0.08,
                            },
                            tooltip=(f"{t('zone_col_id', lang)} {zp['zone_id']}: "
                                     f"{zp.get('area_ha', '—')} {t('ha', lang)}"),
                        ).add_to(outlines)
                    outlines.add_to(m)

                raster_rendered = True
            except Exception as e:
                st.warning(t("raster_error", lang, error=e))

        folium.GeoJson(
            field_geojson,
            name=t("layer_border", lang),
            style_function=lambda feature: {
                "fillColor": "#000000", "fillOpacity": 0.0,
                "color": "#FFD400", "weight": 2.5,
            },
            tooltip=f"{field_name} — {area_ha} {t('ha', lang)}",
        ).add_to(m)

        folium.LayerControl(collapsed=False).add_to(m)
        st_folium(m, width=None, height=440, returned_objects=[])

        if raster_rendered:
            render_ndvi_legend(lang)

        if is_bad:
            st.error(t("map_bad", lang, date=last_point["date"],
                       dev=abs(last_point["deviation_pct"])))
        elif last_point:
            st.success(t("map_ok", lang, date=last_point["date"],
                         dev=last_point["deviation_pct"]))

    with right:
        st.subheader(t("chart_title", lang))

        dates = [p["date"] for p in enriched]
        ndvi_values = [p["ndvi_mean"] for p in enriched]
        ndmi_values = [p["ndmi_mean"] for p in enriched]
        anomaly_pts = [p for p in enriched if p.get("is_anomaly")]
        lowq_pts = [p for p in enriched if p.get("low_quality")]

        fig = go.Figure()

        if mode_history:
            base = [p["baseline_ndvi"] for p in enriched]
            spread = [p.get("baseline_std") or 0 for p in enriched]
            fig.add_trace(go.Scatter(
                x=dates, y=[b + s for b, s in zip(base, spread)], mode="lines",
                line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(
                x=dates, y=[b - s for b, s in zip(base, spread)], mode="lines",
                line=dict(width=0), fill="tonexty", fillcolor="rgba(138,147,140,0.22)",
                name=t("typical_range", lang), hoverinfo="skip"))
            fig.add_trace(go.Scatter(
                x=dates, y=base, mode="lines", name=t("hist_norm_label", lang),
                line=dict(color="#8A938C", dash="dash", width=1.5)))
        else:
            fig.add_hline(y=norm, line_dash="dash", line_color="#8A938C",
                          annotation_text=t("season_norm_label", lang))

        fig.add_trace(go.Scatter(x=dates, y=ndvi_values, mode="lines+markers", name="NDVI",
                                 line=dict(color="#2F6B3A", width=2.5)))
        fig.add_trace(go.Scatter(x=dates, y=ndmi_values, mode="lines+markers", name="NDMI",
                                 line=dict(color="#2B6CA3", width=2)))
        fig.add_trace(go.Scatter(
            x=[p["date"] for p in anomaly_pts], y=[p["ndvi_mean"] for p in anomaly_pts],
            mode="markers", name=t("anomaly_label", lang),
            marker=dict(color="#B4451F", size=13, symbol="x")))
        if lowq_pts:
            fig.add_trace(go.Scatter(
                x=[p["date"] for p in lowq_pts], y=[p["ndvi_mean"] for p in lowq_pts],
                mode="markers", name=t("low_quality_label", lang),
                marker=dict(color="rgba(0,0,0,0)", size=12, symbol="diamond",
                            line=dict(color="#777777", width=2))))

        if prev_timeseries:
            # Точки прошлого сезона ставятся на даты текущего, чтобы кривые
            # сравнивались по этапу сезона. Настоящая дата прошлого года
            # уходит в customdata и показывается в тултипе.
            n = min(len(dates), len(prev_timeseries))
            fig.add_trace(go.Scatter(
                x=dates[:n], y=[prev_timeseries[i]["ndvi_mean"] for i in range(n)],
                mode="lines+markers", name=t("prev_season_label", lang),
                line=dict(color="#C98A1F", dash="dot"),
                customdata=[prev_timeseries[i]["date"] for i in range(n)],
                hovertemplate="%{customdata}<br>NDVI: %{y}<extra></extra>",
            ))

        fig.update_layout(height=440, margin=dict(l=10, r=10, t=10, b=10),
                          plot_bgcolor="#FBFCFA", legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig, use_container_width=True)

# ================== ЗОНЫ ВНУТРИ ПОЛЯ ==================
with tab_zones:
    st.subheader(t("zones_title", lang))
    zstats = st.session_state.get("zone_stats")
    zones = st.session_state.get("zones") or []

    if not show_raster:
        st.info(t("zones_disabled", lang))
    elif not zstats:
        st.warning(t("zones_no_data", lang))
    elif "error" in zstats:
        st.warning(zstats["error"])
    else:
        z1, z2, z3, z4 = st.columns(4)
        with z1:
            agro_card(t("zone_problem_area", lang),
                     f"{zstats.get('problem_area_ha', '—')} {t('ha', lang)}",
                     f"{zstats['problem_share_pct']}% {t('of_field', lang)}", accent=AGRO_ORANGE)
        with z2:
            agro_card(t("zone_mean", lang), str(zstats["field_mean_ndvi"]), accent=AGRO_GREEN)
        with z3:
            agro_card(t("zone_range", lang),
                     f"{zstats['min_ndvi']} – {zstats['max_ndvi']}", accent=AGRO_GREEN)
        with z4:
            cov_accent = AGRO_GREEN if zstats["coverage_pct"] >= 60 else AGRO_RED
            agro_card(t("zone_coverage", lang), f"{zstats['coverage_pct']}%",
                     f"{zstats['valid_pixels']} px", accent=cov_accent)

        st.caption(t("zones_explain", lang, date=zstats["date"],
                     threshold=zstats["threshold_ndvi"], median=zstats["field_median_ndvi"]))
        if zstats.get("non_crop_share_pct", 0) >= 1:
            st.caption(t("zones_non_crop_note", lang, pct=zstats["non_crop_share_pct"],
                         min=NON_CROP_NDVI))

        if zstats["coverage_pct"] < 60:
            st.warning(t("zones_low_coverage", lang, pct=zstats["coverage_pct"]))

        st.markdown(f"**{t('zones_table_title', lang)}**")
        if zones:
            rows = []
            for zone in zones:
                zp = zone["properties"]
                rows.append({
                    t("zone_col_id", lang): zp["zone_id"],
                    t("zone_col_area", lang): zp.get("area_ha"),
                    t("zone_col_share", lang): zp["share_of_field_pct"],
                    t("zone_col_ndvi", lang): zp["mean_ndvi"],
                    t("zone_col_lat", lang): zp["centroid_lat"],
                    t("zone_col_lon", lang): zp["centroid_lon"],
                    t("zone_col_map", lang):
                        f"https://www.google.com/maps?q={zp['centroid_lat']},{zp['centroid_lon']}&t=k",
                })
            st.dataframe(
                pd.DataFrame(rows), hide_index=True, use_container_width=True,
                column_config={t("zone_col_map", lang): st.column_config.LinkColumn(
                    t("zone_col_map", lang), display_text="Google Maps")},
            )
            big = round(sum(z["properties"].get("area_ha", 0) for z in zones), 1)
            st.caption(t("zones_total_note", lang,
                         total=zstats.get("problem_area_ha", "—"), big=big))
        else:
            st.info(t("zones_none", lang))

        st.markdown(t("zones_howto", lang))

# ================== AI-ПРОГНОЗ ==================
with tab_ai:
    reliable_pts = [p for p in enriched if not p.get("low_quality")]
    forecast = forecast_ndvi(reliable_pts, days_ahead=14,
                          history=st.session_state.get("history") or [])
    if "error" in forecast:
        st.info(t("ai_forecast_error", lang))
    else:
        base_last, hist_last = baseline_at(enriched, forecast["last_observation"]["date"])
        base_fc, hist_fc = baseline_at(enriched, forecast["forecast_date"])
        risk = assess_risk(forecast, base_last, base_fc, hist_last and hist_fc)
        trend_key = {"рост": "trend_up", "спад": "trend_down",
                     "стабильно": "trend_flat"}[forecast["trend"]]
        q = forecast["quality"]

        # ---------- 1. График: факт + прогноз + доверительный интервал ----------
        st.subheader(t("ai_chart_title", lang))

        fig = go.Figure()

        actual_dates = [p["date"] for p in enriched]
        actual_values = [p["ndvi_mean"] for p in enriched]
        fig.add_trace(go.Scatter(
            x=actual_dates, y=actual_values, mode="lines+markers",
            name=t("ai_actual_label", lang), line=dict(color=AGRO_GREEN, width=2.5)))

        path = forecast["path"]
        path_dates = [p["date"] for p in path]
        path_values = [p["value"] for p in path]
        path_ci_low = [p["ci_low"] for p in path]
        path_ci_high = [p["ci_high"] for p in path]

        # Серая заливка доверительного интервала — сначала верхняя граница без
        # заливки, затем нижняя с fill="tonexty", это стандартный приём Plotly
        fig.add_trace(go.Scatter(
            x=path_dates, y=path_ci_high, mode="lines", line=dict(width=0),
            showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=path_dates, y=path_ci_low, mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor="rgba(120,120,120,0.20)",
            name=t("ai_ci_label", lang), hoverinfo="skip"))

        fig.add_trace(go.Scatter(
            x=path_dates, y=path_values, mode="lines",
            name=t("ai_forecast_line_label", lang),
            line=dict(color=AGRO_ORANGE, width=2.5, dash="dash")))

        last_obs = forecast["last_observation"]
        fig.add_trace(go.Scatter(
            x=[last_obs["date"]], y=[last_obs["value"]], mode="markers",
            name=t("ai_last_obs_label", lang),
            marker=dict(color=AGRO_GREEN, size=13, symbol="circle",
                        line=dict(color="white", width=2))))
        fig.add_trace(go.Scatter(
            x=[forecast["forecast_date"]], y=[forecast["forecast_ndvi"]], mode="markers",
            name=t("ai_forecast_point_label", lang),
            marker=dict(color=AGRO_ORANGE, size=15, symbol="star",
                        line=dict(color="white", width=2))))

        fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10),
                          plot_bgcolor="#FBFCFA", legend=dict(orientation="h", y=1.15))
        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # ---------- 2. Карточки качества модели ----------
        st.subheader(t("ai_quality_title", lang))
        qc1, qc2, qc3, qc4 = st.columns(4)
        with qc1:
            agro_card(t("metric_r2", lang), f"{q['r2']:.3f}", accent=AGRO_GREEN)
        with qc2:
            agro_card(t("metric_mae", lang), f"{q['mae']:.4f}", accent=AGRO_GREEN)
        with qc3:
            agro_card(t("metric_rmse", lang), f"{q['rmse']:.4f}", accent=AGRO_GREEN)
        with qc4:
            agro_card(t("metric_n", lang), str(q["n"]), accent=AGRO_GREEN)
        st.caption(t("ai_quality_note", lang, n=q["n"]))

        st.divider()
        render_backtest(enriched, st.session_state.get("history") or [],
                        is_real_data, lang)

        st.divider()

        col_trend, col_season = st.columns(2)

        # ---------- 3. Тренд и риск ----------
        with col_trend:
            st.subheader(t("ai_trend_risk_title", lang))
            trend_emoji = {"рост": "📈", "спад": "📉", "стабильно": "➡️"}[forecast["trend"]]
            risk_color = RISK_COLORS[risk["level"]]
            risk_text = t(f"risk_{risk['level']}", lang)
            reco_key = ("risk_reco_needs_history" if risk.get("reason") == "no_history"
                        else f"risk_reco_{risk['level']}")
            reco_text = t(reco_key, lang)

            st.markdown(f"""
            <div class="agro-card" style="--agro-accent: {risk_color};">
                <div style="display:flex; justify-content:space-between; margin-bottom:0.5rem;">
                    <div><b>{t('ai_trend_row', lang)}:</b> {trend_emoji} {t(trend_key, lang)}
                        <span style="color:#6B756D; font-size:0.82rem;">
                            ({forecast['trend_slope_per_day']}/{t('per_day_unit', lang)})
                        </span>
                    </div>
                </div>
                <div style="margin-bottom:0.6rem;">
                    <b>{t('ai_risk_row', lang)}:</b> {agro_chip(risk_text, risk_color)}
                </div>
                <div><b>{t('ai_recommendation_row', lang)}:</b> {reco_text}</div>
            </div>
            """, unsafe_allow_html=True)

        # ---------- 4. Сравнение с прошлым сезоном ----------
        with col_season:
            st.subheader(t("ai_season_compare_title", lang))
            if not prev_timeseries:
                st.info(t("ai_season_compare_missing", lang))
            else:
                comp = compare_seasons(enriched, prev_timeseries)
                if comp is None:
                    st.info(t("ai_season_compare_no_common", lang))
                else:
                    diff_pct = comp["diff_pct"]
                    if diff_pct > 3:
                        verdict, verdict_color = t("ai_season_compare_better", lang), AGRO_GREEN
                    elif diff_pct < -3:
                        verdict, verdict_color = t("ai_season_compare_worse", lang), AGRO_RED
                    else:
                        verdict, verdict_color = t("ai_season_compare_similar", lang), AGRO_ORANGE

                    sc1, sc2 = st.columns(2)
                    with sc1:
                        agro_card(t("ai_season_compare_current", lang, year=comp["cur_year"]),
                                  f"{comp['cur_mean']:.3f}", accent=AGRO_GREEN)
                    with sc2:
                        agro_card(t("ai_season_compare_prev", lang, year=comp["prev_year"]),
                                  f"{comp['prev_mean']:.3f}", accent=AGRO_ORANGE)

                    st.markdown(f"""
                    <div class="agro-card" style="--agro-accent: {verdict_color}; margin-top:0.6rem;">
                        <div class="agro-card-label">{t('ai_season_compare_diff', lang)}</div>
                        <div class="agro-card-value" style="color:{verdict_color};">
                            {'+' if diff_pct >= 0 else ''}{diff_pct}%
                        </div>
                        <div class="agro-card-sub">{verdict}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.caption(t("ai_season_compare_note", lang, n=comp["n_common"]))

    st.divider()

    # ---------- 5. Классификация аномалий + чек-лист рекомендаций ----------
    st.subheader(t("ai_classification_title", lang))
    anomaly_points = [p for p in enriched if p.get("is_anomaly")]

    if not anomaly_points:
        st.success(t("ai_no_anomalies", lang))
    else:
        for p in anomaly_points:
            result = classify_anomaly(p, norm, ndmi_norm)
            best_action = primary_action(result["type"])
            with st.container(border=True):
                col_info, col_checklist = st.columns([1.4, 1])
                with col_info:
                    st.markdown(f"**{p['date']}** — {result['type']}")
                    st.caption(result["explanation"])
                with col_checklist:
                    st.markdown(f"**{t('ai_checklist_title', lang)}**")
                    items_html = ""
                    for item_key in CHECKLIST_ITEMS:
                        cls = "agro-checklist-item primary" if item_key == best_action else "agro-checklist-item"
                        mark = "☑" if item_key == best_action else "☐"
                        items_html += (f'<div class="{cls}">{mark} '
                                       f'{t(f"checklist_{item_key}", lang)}</div>')
                    st.markdown(items_html, unsafe_allow_html=True)

# ================== ОТЧЁТЫ ==================
with tab_reports:
    st.subheader(t("reports_table_title", lang))

    def status_of(p):
        if p.get("low_quality"):
            return t("status_lowq", lang)
        return t("status_anomaly", lang) if p.get("is_anomaly") else t("status_ok", lang)

    status_flag_col = t("col_flag", lang)
    table = pd.DataFrame([{
        t("col_date", lang): p["date"],
        t("col_ndvi", lang): p["ndvi_mean"],
        t("col_baseline", lang): p.get("baseline_ndvi"),
        t("col_ndmi", lang): p["ndmi_mean"],
        t("col_dev", lang): p.get("deviation_pct"),
        t("col_share", lang): p.get("data_share_pct"),
        status_flag_col: status_of(p),
    } for p in enriched])

    status_bg = {
        t("status_anomaly", lang): "background-color: #FDEAEA; color: #B3261E; font-weight: 600;",
        t("status_ok", lang): "background-color: #E9F5EA; color: #1E6B24; font-weight: 600;",
        t("status_lowq", lang): "background-color: #F1F1F1; color: #6B6B6B; font-weight: 600;",
    }

    def _style_status(val):
        return status_bg.get(val, "")

    try:
        styled_table = table.style.map(_style_status, subset=[status_flag_col])
    except AttributeError:
        # pandas <2.1 — .map у Styler появился в 2.1, до этого был .applymap
        styled_table = table.style.applymap(_style_status, subset=[status_flag_col])
    st.dataframe(styled_table, hide_index=True, use_container_width=True)

    st.subheader(t("reports_export_title", lang))
    exp_col1, exp_col2, exp_col3 = st.columns(3)

    zone_stats_for_report = st.session_state.get("zone_stats")
    zones_for_report = st.session_state.get("zones") or []

    with exp_col1:
        st.download_button(t("download_csv", lang), data=export_csv(enriched),
                           file_name="ndvi_report.csv", mime="text/csv",
                           use_container_width=True)

    with exp_col2:
        st.download_button(t("download_geojson", lang),
                           data=export_geojson(field_geojson, summary, field_name,
                                               zone_stats=zone_stats_for_report,
                                               area_ha=area_ha, zones=zones_for_report),
                           file_name="field_report.geojson", mime="application/geo+json",
                           use_container_width=True)

    with exp_col3:
        try:
            pdf_bytes = export_pdf_report(field_name, summary, norm, enriched,
                                          zone_stats=zone_stats_for_report,
                                          area_ha=area_ha, zones=zones_for_report)
            st.download_button(t("download_pdf", lang), data=pdf_bytes,
                               file_name="ndvi_report.pdf", mime="application/pdf",
                               use_container_width=True)
        except Exception as e:
            st.warning(t("pdf_unavailable", lang, error=e))