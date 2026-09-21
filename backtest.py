"""
Проверка прогноза NDVI на данных, которых модель не видела (backtest).

Метрики R²/MAE/RMSE внутри forecast_ndvi() считаются на тех же 5 точках, по
которым построена линия, и показывают только, насколько хорошо прямая легла на
эти точки. Они ничего не говорят о качестве прогноза. Здесь проверка честная —
скользящее окно с «прогнозом в будущее»:

    для каждой даты t каждого сезона:
        обучаем модель на 5 надёжных точках, заканчивающихся на t;
        прогнозируем NDVI на t + horizon дней;
        сравниваем с фактом (линейная интерполяция между соседними снимками,
        потому что декадный ряд редко попадает ровно на t + 14).

Модели сравниваются между собой. Прогноз, который не лучше «завтра будет как
сегодня», ценности не имеет, поэтому обязательны базовые модели:

    persistence       — NDVI не изменится (последнее наблюдение);
    climatology       — NDVI будет равен норме этой декады по другим сезонам;
    persistence_norm  — последнее наблюдение плюс типичное изменение нормы
                        за горизонт (учитывает сезонный рост и спад);
    linear            — линейный тренд последних 5 точек (модель приложения);
    blend             — среднее linear и persistence_norm.

Норма для каждого сезона строится по ДРУГИМ сезонам (leave-one-out): иначе
сезон сравнивался бы сам с собой.
"""

from datetime import datetime, timedelta
from math import sqrt

from analysis import build_baseline, decade_key, is_low_quality
from ai_analysis import forecast_ndvi

MODEL_ORDER = ["persistence", "climatology", "persistence_norm", "linear", "blend"]
MIN_OTHER_SEASONS = 2       # столько прошлых сезонов нужно для нормы
MAX_BRACKET_GAP_DAYS = 25   # интерполировать факт можно только между близкими снимками
MAX_WINDOW_SPAN_DAYS = 60   # окно из 5 точек не должно растягиваться на пол-сезона


def _d(date_str: str) -> datetime:
    return datetime.strptime(date_str, "%Y-%m-%d")


def _reliable(series: list[dict]) -> list[dict]:
    pts = [p for p in series if p.get("ndvi_mean") is not None and not is_low_quality(p)]
    return sorted(pts, key=lambda p: p["date"])


def interpolate_actual(points: list[dict], target: datetime) -> float | None:
    """Факт на дату target: линейно между двумя соседними надёжными снимками."""
    before = [p for p in points if _d(p["date"]) <= target]
    after = [p for p in points if _d(p["date"]) >= target]
    if not before or not after:
        return None
    a, b = before[-1], after[0]
    da, db = _d(a["date"]), _d(b["date"])
    if da == db:
        return a["ndvi_mean"]
    if (db - da).days > MAX_BRACKET_GAP_DAYS:
        return None
    w = (target - da).days / (db - da).days
    return a["ndvi_mean"] + w * (b["ndvi_mean"] - a["ndvi_mean"])


def _norm_at(baseline: dict, date_str: str) -> float | None:
    entry = baseline.get(decade_key(date_str))
    return entry["mean"] if entry and entry["n"] >= MIN_OTHER_SEASONS else None


def run_backtest(seasons: list[list[dict]], horizon_days: int = 14,
                 window: int = 5) -> dict:
    """
    Скользящий backtest по всем сезонам.

    seasons: список рядов [{date, ndvi_mean, valid_pixels, total_pixels, ...}, ...],
             текущий сезон вместе с прошлыми.

    Возвращает {"error": ...} либо
        {"horizon_days", "window", "n_samples", "n_seasons", "has_norm",
         "models": {имя: {mae, rmse, bias, skill_vs_persistence_pct}},
         "samples": [{season, origin, target, actual, <модель>: прогноз, ...}]}
    """
    prepared = [_reliable(s) for s in seasons]
    prepared = [s for s in prepared if len(s) >= window + 1]
    if not prepared:
        return {"error": "too_short"}

    samples = []
    for idx, pts in enumerate(prepared):
        others = [prepared[j] for j in range(len(prepared)) if j != idx]
        has_norm = len(others) >= MIN_OTHER_SEASONS
        baseline = build_baseline(others) if has_norm else {}

        for i in range(window - 1, len(pts)):
            train = pts[i - window + 1: i + 1]
            origin = _d(train[-1]["date"])
            if (origin - _d(train[0]["date"])).days > MAX_WINDOW_SPAN_DAYS:
                continue
            target = origin + timedelta(days=horizon_days)
            actual = interpolate_actual(pts, target)
            if actual is None:
                continue

            fc = forecast_ndvi(train, days_ahead=horizon_days, use_last_n=window)
            if "error" in fc:
                continue

            last = train[-1]["ndvi_mean"]
            row = {
                "season": pts[0]["date"][:4],
                "origin": train[-1]["date"],
                "target": target.strftime("%Y-%m-%d"),
                "actual": actual,
                "persistence": last,
                "linear": fc["forecast_ndvi"],
            }
            if has_norm:
                clim_t = _norm_at(baseline, row["target"])
                clim_o = _norm_at(baseline, row["origin"])
                if clim_t is None or clim_o is None:
                    continue  # для этой даты нет нормы — образец не участвует ни в одной модели
                row["climatology"] = clim_t
                row["persistence_norm"] = max(0.05, min(0.95, last + (clim_t - clim_o)))
                row["blend"] = 0.5 * row["linear"] + 0.5 * row["persistence_norm"]
            samples.append(row)

    if not samples:
        return {"error": "no_samples"}

    has_norm = "climatology" in samples[0]
    names = [m for m in MODEL_ORDER if (has_norm or m in ("persistence", "linear"))]

    models = {}
    for name in names:
        errs = [s[name] - s["actual"] for s in samples]
        n = len(errs)
        models[name] = {
            "mae": sum(abs(e) for e in errs) / n,
            "rmse": sqrt(sum(e * e for e in errs) / n),
            "bias": sum(errs) / n,
        }
    base_mae = models["persistence"]["mae"]
    for name, m in models.items():
        m["skill_vs_persistence_pct"] = (
            round(100 * (1 - m["mae"] / base_mae), 1) if base_mae > 0 else 0.0)

    return {
        "horizon_days": horizon_days,
        "window": window,
        "n_samples": len(samples),
        "n_seasons": len(prepared),
        "has_norm": has_norm,
        "best_model": min(models, key=lambda k: models[k]["mae"]),
        "models": models,
        "samples": samples,
    }
