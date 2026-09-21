"""
Тесты расчётной части (без сети и без Streamlit).

    pip install pytest
    pytest -q
"""

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai_analysis import assess_risk, classify_anomaly, forecast_ndvi
from analysis import (build_baseline, compare_seasons, decade_key,
                      detect_anomalies, is_low_quality)
from backtest import interpolate_actual, run_backtest
from config import DEFAULT_FIELD_GEOJSON
from data_cache import load_snapshot, load_snapshot_raster, save_snapshot, snapshot_available
from i18n import TRANSLATIONS
from mock_data import generate_mock_timeseries
from raster import (extract_problem_zones, field_area_ha, mock_ndvi_raster,
                    normalize_field_geometry, zone_statistics)


def make_series(year, values, start_day=1, step=10, valid=900, total=900):
    """Ряд с шагом step дней от 1 апреля."""
    from datetime import date, timedelta
    d0 = date(year, 4, start_day)
    return [{"date": (d0 + timedelta(days=step * i)).isoformat(), "ndvi_mean": v,
             "ndmi_mean": v * 0.5, "ndvi_std": 0.03,
             "valid_pixels": valid, "total_pixels": total} for i, v in enumerate(values)]


# ---------- analysis ----------

def test_decade_key_consecutive_and_leap_day():
    assert decade_key("2026-04-11") == decade_key("2026-04-19")
    assert decade_key("2026-04-21") == decade_key("2026-04-11") + 1
    assert decade_key("2024-02-29") == decade_key("2024-02-28")


def test_anomaly_flagged_and_low_quality_never_flagged():
    history = [make_series(2025, [0.4, 0.5, 0.6, 0.7, 0.7, 0.6]),
               make_series(2024, [0.4, 0.5, 0.6, 0.7, 0.7, 0.6])]
    cur = make_series(2026, [0.4, 0.5, 0.35, 0.7, 0.3, 0.6])
    cur[4]["valid_pixels"] = 100  # 11% покрытия
    enriched, _ = detect_anomalies(cur, history=history)
    assert enriched[2]["is_anomaly"]
    assert enriched[4]["low_quality"] and not enriched[4]["is_anomaly"]
    assert not enriched[0]["is_anomaly"]
    assert enriched[2]["baseline_kind"] == "history"


def test_small_absolute_drop_is_not_anomaly():
    history = [make_series(2025, [0.06] * 5), make_series(2024, [0.06] * 5)]
    cur = make_series(2026, [0.03] * 5)  # -50%, но падение 0.03 < MIN_ABS_DROP
    enriched, _ = detect_anomalies(cur, history=history)
    assert not any(p["is_anomaly"] for p in enriched)


def test_baseline_ignores_low_quality_points():
    good = make_series(2025, [0.5, 0.6])
    bad = make_series(2024, [0.1, 0.1], valid=50)
    base = build_baseline([good, bad])
    assert all(v["n"] == 1 for v in base.values())


def test_compare_seasons_needs_common_decades():
    a = make_series(2026, [0.5, 0.6, 0.7, 0.6])
    b = make_series(2025, [0.4, 0.5, 0.6, 0.5])
    res = compare_seasons(a, b)
    assert res and res["diff_pct"] > 0
    assert compare_seasons(a[:2], b[:2]) is None


# ---------- ai_analysis ----------

def test_forecast_needs_three_points():
    assert "error" in forecast_ndvi(make_series(2026, [0.5, 0.6]))


def test_forecast_is_clamped_and_interval_ordered():
    falling = make_series(2026, [0.5, 0.4, 0.3, 0.2, 0.1])
    fc = forecast_ndvi(falling, days_ahead=14)
    assert fc["trend"] == "спад"
    assert 0.05 <= fc["forecast_ndvi"] <= 0.95
    for p in fc["path"]:
        assert 0.05 <= p["ci_low"] <= p["value"] <= p["ci_high"] <= 0.95


def test_forecast_recovers_linear_trend():
    rising = make_series(2026, [0.30, 0.34, 0.38, 0.42, 0.46])  # +0.004/день
    fc = forecast_ndvi(rising, days_ahead=10)
    assert abs(fc["forecast_ndvi"] - 0.50) < 1e-3
    assert abs(fc["trend_slope_per_day"] - 0.004) < 1e-4


def test_risk_unknown_without_history():
    fc = forecast_ndvi(make_series(2026, [0.6, 0.5, 0.4, 0.3, 0.25]))
    assert assess_risk(fc, 0.6, 0.6, baseline_is_history=False)["level"] == "unknown"


def test_risk_high_when_forecast_far_below_norm():
    fc = forecast_ndvi(make_series(2026, [0.7, 0.6, 0.5, 0.4, 0.3]))
    assert assess_risk(fc, 0.7, 0.7, baseline_is_history=True)["level"] == "high"


def test_classify_drought_vs_disease():
    drought = {"ndvi_mean": 0.4, "ndmi_mean": 0.15, "ndvi_std": 0.03}
    disease = {"ndvi_mean": 0.4, "ndmi_mean": 0.30, "ndvi_std": 0.03}
    assert classify_anomaly(drought, 0.6, 0.3)["type"] == "Засуха / суховей"
    assert classify_anomaly(disease, 0.6, 0.3)["type"] == "Возможная болезнь/вредитель"


# ---------- backtest ----------

def test_interpolate_actual():
    pts = make_series(2026, [0.4, 0.6])  # 1 и 11 апреля
    assert abs(interpolate_actual(pts, __import__("datetime").datetime(2026, 4, 6)) - 0.5) < 1e-9
    assert interpolate_actual(pts, __import__("datetime").datetime(2026, 5, 1)) is None


def test_backtest_perfect_on_linear_series():
    seasons = [make_series(y, [0.20 + 0.03 * i for i in range(12)]) for y in (2026, 2025, 2024)]
    res = run_backtest(seasons, horizon_days=10)
    assert res["models"]["linear"]["mae"] < 1e-6
    assert res["models"]["persistence"]["mae"] > 0.02


def test_backtest_without_history_has_only_two_models():
    res = run_backtest([make_series(2026, [0.2 + 0.03 * i for i in range(12)])])
    assert set(res["models"]) == {"persistence", "linear"}
    assert not res["has_norm"]


def test_backtest_norm_excludes_own_season():
    """Норма считается по другим сезонам: сезон с аномальным уровнем не должен
    'подсматривать' самого себя."""
    a = make_series(2026, [0.9] * 12)
    others = [make_series(2025, [0.3] * 12), make_series(2024, [0.3] * 12)]
    res = run_backtest([a] + others, horizon_days=10)
    own = [s for s in res["samples"] if s["season"] == "2026"]
    assert own and all(abs(s["climatology"] - 0.3) < 1e-9 for s in own)


def test_backtest_too_short():
    assert run_backtest([make_series(2026, [0.4, 0.5, 0.6])]) == {"error": "too_short"}


# ---------- data_cache ----------

def test_snapshot_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        assert not snapshot_available(d) and load_snapshot(d) is None
        ndvi = np.random.default_rng(0).uniform(0.2, 0.8, (20, 30)).astype(np.float32)
        mask = np.ones((20, 30), dtype=np.float32)
        ts = make_series(2026, [0.4, 0.5, 0.6])
        save_snapshot(DEFAULT_FIELD_GEOJSON, "2026-04-01", "2026-09-15", ts,
                      [make_series(2025, [0.4, 0.5])], {"2026-04-21": (ndvi, mask)}, d)
        snap = load_snapshot(d)
        assert snap["timeseries"] == ts and len(snap["history"]) == 1
        assert snap["meta"]["raster_dates"] == ["2026-04-21"]
        got = load_snapshot_raster("2026-04-21", d)
        assert np.allclose(got[0], ndvi) and got[1].shape == mask.shape
        assert load_snapshot_raster("2026-05-01", d) is None


def test_broken_snapshot_returns_none():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "meta.json").write_text("{not json", "utf-8")
        (d / "series.json").write_text("{}", "utf-8")
        assert load_snapshot(d) is None


# ---------- raster ----------

def test_default_field_area_about_300_ha():
    assert 280 < field_area_ha(DEFAULT_FIELD_GEOJSON) < 320


def test_normalize_feature_collection():
    poly = DEFAULT_FIELD_GEOJSON
    fc = {"type": "FeatureCollection",
          "features": [{"type": "Feature", "properties": {}, "geometry": poly}]}
    assert normalize_field_geometry(fc)["type"] == "Polygon"


def test_zones_found_in_mock_raster_with_defect():
    ndvi, mask = mock_ndvi_raster(DEFAULT_FIELD_GEOJSON, base=0.6, strength=0.25)
    area = field_area_ha(DEFAULT_FIELD_GEOJSON)
    st = zone_statistics(ndvi, mask, field_area=area)
    assert "error" not in st and 0 <= st["problem_share_pct"] <= 100
    zones = extract_problem_zones(ndvi, mask, st["threshold_ndvi"], DEFAULT_FIELD_GEOJSON, area)
    assert len(zones) >= 1
    assert all(z["properties"]["area_ha"] > 0 for z in zones)
    assert st["problem_area_ha"] > 0


def test_uniform_healthy_field_has_no_problem_pixels():
    rng = np.random.default_rng(1)
    ndvi = (0.65 + rng.normal(0, 0.01, (60, 80))).astype(np.float32)
    st = zone_statistics(ndvi, np.ones_like(ndvi), field_area=100)
    assert st["problem_pixels"] == 0


def test_no_valid_pixels_returns_error():
    ndvi = np.full((10, 10), 0.5, dtype=np.float32)
    assert "error" in zone_statistics(ndvi, np.zeros_like(ndvi))


# ---------- i18n / mock ----------

def test_translation_keys_match():
    ru, kk = set(TRANSLATIONS["ru"]), set(TRANSLATIONS["kk"])
    assert not (ru ^ kk), sorted(ru ^ kk)


def test_mock_series_has_low_quality_periods():
    ts = generate_mock_timeseries("2026-04-01", "2026-09-15")
    assert sum(is_low_quality(p) for p in ts) == 2
