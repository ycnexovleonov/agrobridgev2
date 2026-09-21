"""
Аналитика поверх временного ряда NDVI/NDMI.

1. forecast_ndvi() — прогноз NDVI на N дней вперёд по линейному тренду
   последних наблюдений, с доверительным интервалом по остаткам модели.
2. classify_anomaly() — определение вероятной причины аномалии по
   совместному поведению NDVI и NDMI.

Выбор простой регрессии вместо ARIMA/LSTM продиктован объёмом данных:
сезон даёт 15-18 точек по декадам, на таком ряду сложные модели дают
неустойчивый прогноз.
"""

import numpy as np
from datetime import datetime, timedelta
from scipy import stats as scipy_stats

from config import MIN_ABS_DROP

# Физические пределы NDVI для поля: голая почва / стерня и плотный посев
NDVI_FLOOR = 0.05
NDVI_CEIL = 0.95


def forecast_ndvi(timeseries: list[dict], days_ahead: int = 14,
                  use_last_n: int = 5, history: list[list[dict]] | None = None) -> dict:
    """
    Прогноз NDVI на days_ahead дней вперёд.

    Основная модель — норма декады по прошлым сезонам (climatology).
    Backtest показал, что она точнее линейного тренда:
        линейный тренд: MAE = 0.2033 (выигрыш 6.8%)
        норма декады:   MAE = 0.1217 (выигрыш 44.2%)

    Если истории нет — fallback на линейную регрессию.
    """
    if len(timeseries) < 3:
        return {"error": "Недостаточно данных для прогноза (нужно минимум 3 точки)"}

    from analysis import build_baseline, decade_key
    from datetime import timedelta

    recent = timeseries[-use_last_n:] if len(timeseries) >= use_last_n else timeseries
    n = len(recent)

    dates = [datetime.strptime(p["date"], "%Y-%m-%d") for p in recent]
    values = np.array([p["ndvi_mean"] for p in recent], dtype=float)

    day0 = dates[0]
    x = np.array([(d - day0).days for d in dates], dtype=float)

    slope, intercept = np.polyfit(x, values, deg=1)
    fitted = slope * x + intercept
    residuals = values - fitted

    dof = max(n - 2, 1)
    ss_res = float(np.sum(residuals ** 2))
    residual_std = float(np.sqrt(ss_res / dof))

    x_mean = float(x.mean())
    sxx = float(np.sum((x - x_mean) ** 2))
    t_crit = float(scipy_stats.t.ppf(0.95, dof))

    def predict_linear(x0: float):
        value = float(slope * x0 + intercept)
        se = residual_std * np.sqrt(1 + 1 / n + ((x0 - x_mean) ** 2 / sxx if sxx > 0 else 0))
        margin = t_crit * se
        return value, margin

    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(ss_res / n))
    ss_tot = float(np.sum((values - values.mean()) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    last_date = dates[-1]
    last_x = x[-1]
    last_actual = float(values[-1])

    baseline = build_baseline(history) if history else {}

    path = [{
        "date": last_date.strftime("%Y-%m-%d"), "value": round(last_actual, 4),
        "ci_low": round(last_actual, 4), "ci_high": round(last_actual, 4),
    }]

    step = 2
    horizons = list(range(step, days_ahead, step)) + [days_ahead]

    for h in horizons:
        d = last_date + timedelta(days=h)
        date_str = d.strftime("%Y-%m-%d")

        if baseline:
            key = decade_key(date_str)
            entry = baseline.get(key)
            if entry and entry.get("n", 0) >= 2:
                value = entry["mean"]
                sigma = entry.get("std", 0.05) or 0.05
                margin = 1.645 * sigma
            else:
                x0 = last_x + h
                value, margin = predict_linear(x0)
        else:
            x0 = last_x + h
            value, margin = predict_linear(x0)

        value = max(NDVI_FLOOR, min(NDVI_CEIL, value))
        path.append({
            "date": date_str,
            "value": round(value, 4),
            "ci_low": round(float(max(NDVI_FLOOR, value - margin)), 4),
            "ci_high": round(float(min(NDVI_CEIL, value + margin)), 4),
        })

    forecast_value = path[-1]["value"]
    ci_low, ci_high = path[-1]["ci_low"], path[-1]["ci_high"]

    if slope > 0.002:
        trend = "рост"
    elif slope < -0.002:
        trend = "спад"
    else:
        trend = "стабильно"

    return {
        "forecast_date": path[-1]["date"],
        "forecast_ndvi": forecast_value,
        "confidence_interval": (ci_low, ci_high),
        "trend": trend,
        "trend_slope_per_day": round(float(slope), 5),
        "n_points_used": n,
        "path": path,
        "last_observation": {"date": last_date.strftime("%Y-%m-%d"), "value": round(last_actual, 4)},
        "quality": {
            "r2": round(r2, 3),
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "n": n,
        },
        "model": "climatology" if baseline else "linear",
    }

def assess_risk(forecast: dict, base_last: float | None, base_forecast: float | None,
                baseline_is_history: bool) -> dict:
    """
    Уровень риска по прогнозу — относительно нормы ТОЙ ЖЕ ДЕКАДЫ.

    Сравнивать прогноз со средней нормой за весь сезон нельзя: в конце сезона
    NDVI закономерно падает, и такое сравнение всегда давало бы «высокий риск».
    Поэтому учитываются два показателя:
      deviation     — насколько прогноз ниже нормы на дату прогноза;
      excess_change — насколько NDVI падает быстрее, чем падает сама норма
                      (нормальный сезонный спад риском не считается).

    Если нормы по истории нет, уровень не оценивается: относительно среднего
    по сезону оценка была бы ненадёжной.
    """
    if "error" in forecast or not base_last or not base_forecast:
        return {"level": "unknown", "reason": "no_data"}
    if not baseline_is_history:
        return {"level": "unknown", "reason": "no_history"}

    last_value = forecast["last_observation"]["value"]
    predicted = forecast["forecast_ndvi"]

    deviation = (predicted - base_forecast) / base_forecast
    abs_drop = base_forecast - predicted
    excess_change = (predicted - last_value) - (base_forecast - base_last)

    significant = abs_drop >= MIN_ABS_DROP

    if significant and (deviation <= -0.25 or (deviation <= -0.15 and excess_change <= -0.05)):
        level = "high"
    elif (significant and deviation <= -0.10) or excess_change <= -0.08:
        level = "medium"
    else:
        level = "low"

    return {"level": level, "deviation_pct": round(deviation * 100, 1),
            "excess_change": round(excess_change, 3)}


CHECKLIST_ITEMS = ["moisture", "inspect", "agronomist"]

_PRIMARY_ACTION_BY_TYPE = {
    "Засуха / суховей": "moisture",
    "Возможная болезнь/вредитель": "inspect",
    "Засорённость / неоднородность": "inspect",
    "Неопределённая аномалия": "agronomist",
}


def primary_action(anomaly_type: str) -> str:
    """Какой из трёх пунктов чек-листа выделить как приоритетный для этого типа аномалии."""
    return _PRIMARY_ACTION_BY_TYPE.get(anomaly_type, "agronomist")


def classify_anomaly(point: dict, ndvi_norm: float, ndmi_norm: float) -> dict:
    """
    Классифицирует ВЕРОЯТНУЮ причину аномалии по паттерну NDVI/NDMI.

    Логика опирается на физический смысл индексов:
    - NDVI низкий И NDMI низкий -> растению не хватает влаги -> ЗАСУХА/СУХОВЕЙ
      (индекс влажности тоже просел — значит дело именно в воде)
    - NDVI низкий, но NDMI В НОРМЕ -> влага есть, а зелёная масса всё равно
      просела -> ВОЗМОЖНАЯ БОЛЕЗНЬ/ВРЕДИТЕЛЬ (растение "болеет" не от засухи)
    - NDVI умеренно низкий, много "шума" (высокий std) -> ЗАСОРЁННОСТЬ /
      неоднородность посева (смесь сорняков и культуры даёт пятнистую,
      нестабильную картину индекса внутри контура)
    - иначе -> НЕОПРЕДЕЛЁННАЯ АНОМАЛИЯ (нужен визуальный осмотр)

    Это экспертное правило поверх физического смысла индексов, а не
    обученная модель: результат проверяем агрономом в поле.
    """
    ndvi = point.get("ndvi_mean")
    ndmi = point.get("ndmi_mean")
    ndvi_std = point.get("ndvi_std", 0)

    # Если у точки есть норма своей декады (история прошлых сезонов), сравниваем
    # с ней; единая норма по сезону остаётся запасным вариантом.
    ndvi_norm = point.get("baseline_ndvi") or ndvi_norm
    ndmi_norm = point.get("baseline_ndmi") or ndmi_norm

    if ndvi is None:
        return {"type": "нет данных", "explanation": "NDVI не рассчитан для этого периода"}

    ndvi_deviation = (ndvi - ndvi_norm) / ndvi_norm if ndvi_norm else 0
    ndmi_deviation = (ndmi - ndmi_norm) / ndmi_norm if (ndmi is not None and ndmi_norm) else 0

    if ndvi_deviation <= -0.15 and ndmi_deviation <= -0.15:
        return {
            "type": "Засуха / суховей",
            "explanation": f"Оба индекса просели (NDVI {ndvi_deviation*100:.0f}%, "
                            f"NDMI {ndmi_deviation*100:.0f}%) — признак дефицита влаги.",
        }

    if ndvi_deviation <= -0.15 and ndmi_deviation > -0.10:
        return {
            "type": "Возможная болезнь/вредитель",
            "explanation": f"NDVI просел ({ndvi_deviation*100:.0f}%), но влажность (NDMI) "
                            f"в норме — вода есть, а зелёная масса всё равно снижена.",
        }

    if ndvi_deviation <= -0.10 and ndvi_std and ndvi_std > 0.08:
        return {
            "type": "Засорённость / неоднородность",
            "explanation": f"Умеренное снижение NDVI при высоком разбросе внутри поля "
                            f"(std={ndvi_std}) — типично для пятнистого засорения сорняками.",
        }

    if ndvi_deviation <= -0.10:
        return {
            "type": "Неопределённая аномалия",
            "explanation": "Есть отклонение от нормы, но паттерн не однозначен — "
                            "рекомендуется визуальный осмотр участка агрономом.",
        }

    return {"type": "В пределах нормы", "explanation": "Существенных отклонений не выявлено."}
