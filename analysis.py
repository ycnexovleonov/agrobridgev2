"""
Выделение отклонений от нормы во временном ряду индексов.

Есть два способа задать норму:

  1. По истории поля — среднее значение той же декады в прошлых сезонах.
     Это корректная база сравнения: она учитывает фенологию, то есть то, что
     ранней весной и после уборки NDVI низкий у любого здорового поля.

  2. По текущему сезону — среднее по всему ряду. Запасной вариант, когда
     истории нет. У него есть известный недостаток: весенние и осенние точки
     почти всегда оказываются ниже среднего, поэтому такая норма сигнализирует
     о проблемах чаще, чем они есть на самом деле.

Точка считается аномалией, если одновременно:
  - отклонение от нормы ниже порога ANOMALY_THRESHOLD (в долях от нормы);
  - абсолютное падение не меньше MIN_ABS_DROP (иначе шум у нулевых значений
    даёт ложные срабатывания);
  - покрытие снимком не ниже MIN_VALID_SHARE (иначе среднее по нескольким
    пикселям вне облака ничего не говорит о поле).
"""

from datetime import datetime
from statistics import mean, pstdev

from config import (ANOMALY_THRESHOLD, MIN_ABS_DROP, MIN_HISTORY_SEASONS,
                    MIN_VALID_SHARE)


def shift_year(date_str: str, delta_years: int) -> str:
    """Сдвигает дату 'YYYY-MM-DD' на delta_years лет: тот же период в другом сезоне."""
    d = datetime.strptime(date_str, "%Y-%m-%d")
    try:
        shifted = d.replace(year=d.year + delta_years)
    except ValueError:  # 29 февраля в невисокосном году
        shifted = d.replace(year=d.year + delta_years, day=28)
    return shifted.strftime("%Y-%m-%d")


def compute_seasonal_norm(timeseries: list[dict], field: str = "ndvi_mean") -> float:
    """Среднее значение индекса по ряду — норма по текущему сезону."""
    values = [point[field] for point in timeseries if point.get(field) is not None]
    return round(mean(values), 4) if values else 0.0


def decade_key(date_str: str) -> int:
    """
    Номер декады в году, по которому выравниваются сезоны.

    Дата переводится в невисокосный год, чтобы 29 февраля не сдвигало
    границы, и делится на 10 дней. Ряд с шагом 10 дней при этом даёт
    последовательные целые ключи — а значит, пропущенная из-за облачности
    точка не сдвигает остальные, как это было бы при сопоставлении по
    порядковому номеру.
    """
    d = datetime.strptime(date_str, "%Y-%m-%d")
    day = min(d.day, 28) if d.month == 2 else d.day
    return datetime(2001, d.month, day).timetuple().tm_yday // 10


def data_share(point: dict) -> float | None:
    """Доля валидных пикселей в периоде (0..1) или None, если данных о покрытии нет."""
    total = point.get("total_pixels")
    valid = point.get("valid_pixels")
    if not total or valid is None:
        return None
    return valid / total


def is_low_quality(point: dict) -> bool:
    share = data_share(point)
    return share is not None and share < MIN_VALID_SHARE


def build_baseline(history: list[list[dict]], field: str = "ndvi_mean") -> dict[int, dict]:
    """
    Норма по декадам из прошлых сезонов: {ключ_декады: {mean, std, n}}.

    Внутри одного сезона значения одной декады усредняются, затем среднее
    берётся по сезонам. Точки с низким покрытием снимком в норму не идут.
    """
    per_bucket: dict[int, list[float]] = {}

    for season in history:
        season_bucket: dict[int, list[float]] = {}
        for p in season:
            value = p.get(field)
            if value is None or is_low_quality(p):
                continue
            season_bucket.setdefault(decade_key(p["date"]), []).append(value)
        for key, values in season_bucket.items():
            per_bucket.setdefault(key, []).append(mean(values))

    baseline = {}
    for key, values in per_bucket.items():
        baseline[key] = {
            "mean": mean(values),
            "std": pstdev(values) if len(values) > 1 else 0.0,
            "n": len(values),
        }
    return baseline


def detect_anomalies(timeseries: list[dict], field: str = "ndvi_mean",
                     threshold: float = ANOMALY_THRESHOLD,
                     history: list[list[dict]] | None = None) -> tuple[list[dict], float]:
    """
    Добавляет к каждой точке ряда норму, отклонение и флаги.

    Поля, которые появляются в точках:
        baseline_ndvi, baseline_ndmi, baseline_std — норма для этой декады
        baseline_kind — "history" (по прошлым сезонам) или "season" (по текущему)
        deviation_pct — отклонение от нормы в процентах
        is_anomaly    — точка проблемная
        low_quality   — покрытие снимком ниже MIN_VALID_SHARE

    Возвращает: (обогащённый ряд, единая норма для карточки в интерфейсе).
    """
    good = [p for p in timeseries if not is_low_quality(p)]
    season_norm = compute_seasonal_norm(good or timeseries, field)
    season_ndmi_norm = compute_seasonal_norm(good or timeseries, "ndmi_mean")

    ndvi_base = build_baseline(history, "ndvi_mean") if history else {}
    ndmi_base = build_baseline(history, "ndmi_mean") if history else {}

    enriched = []
    used_norms = []

    for point in timeseries:
        value = point.get(field)
        key = decade_key(point["date"])
        hist = ndvi_base.get(key)

        if hist and hist["n"] >= MIN_HISTORY_SEASONS:
            kind = "history"
            base_ndvi = hist["mean"]
            base_std = hist["std"]
            ndmi_hist = ndmi_base.get(key)
            base_ndmi = ndmi_hist["mean"] if ndmi_hist else season_ndmi_norm
        else:
            kind = "season"
            base_ndvi = season_norm
            base_std = 0.0
            base_ndmi = season_ndmi_norm

        low_quality = is_low_quality(point)
        deviation = (value - base_ndvi) / base_ndvi if (value is not None and base_ndvi) else 0.0
        drop = (base_ndvi - value) if value is not None else 0.0

        is_anomaly = (
            not low_quality
            and deviation <= threshold
            and drop >= MIN_ABS_DROP
        )

        if not low_quality:
            used_norms.append(base_ndvi)

        share = data_share(point)
        enriched.append({
            **point,
            "baseline_ndvi": round(base_ndvi, 4),
            "baseline_ndmi": round(base_ndmi, 4),
            "baseline_std": round(base_std, 4),
            "baseline_kind": kind,
            "deviation_pct": round(deviation * 100, 1),
            "data_share_pct": round(share * 100) if share is not None else None,
            "low_quality": low_quality,
            "is_anomaly": is_anomaly,
        })

    norm = round(mean(used_norms), 4) if used_norms else season_norm
    return enriched, norm


def norm_mode(enriched: list[dict]) -> str:
    """'history', если норма по прошлым сезонам построена для большей части ряда, иначе 'season'."""
    if not enriched:
        return "season"
    with_history = sum(1 for p in enriched if p.get("baseline_kind") == "history")
    return "history" if with_history >= 0.6 * len(enriched) else "season"


def summarize(enriched_timeseries: list[dict]) -> dict:
    """Сводка для карточек интерфейса и отчёта."""
    total = len(enriched_timeseries)
    anomalies = [p for p in enriched_timeseries if p.get("is_anomaly")]
    reliable = [p for p in enriched_timeseries if not p.get("low_quality")]

    return {
        "total_periods": total,
        "anomaly_periods": len(anomalies),
        "anomaly_share_pct": round(100 * len(anomalies) / total, 1) if total else 0,
        "low_quality_periods": total - len(reliable),
        "worst_point": min(reliable, key=lambda p: p.get("deviation_pct", 0)) if reliable else None,
        "latest_point": enriched_timeseries[-1] if enriched_timeseries else None,
        "norm_mode": norm_mode(enriched_timeseries),
    }


def baseline_at(enriched: list[dict], date_str: str) -> tuple[float | None, bool]:
    """
    Норма NDVI для произвольной даты и признак «норма построена по истории».

    Для даты внутри ряда берётся норма её декады. Для даты за краем ряда
    (прогноз уходит на 14 дней вперёд, за конец сезона) норма линейно
    продолжается по двум последним декадам: плоское продолжение занижало бы
    ожидаемый спад в конце сезона.
    """
    if not enriched:
        return None, False

    by_key = {}
    for p in enriched:
        by_key.setdefault(decade_key(p["date"]), p)
    keys = sorted(by_key)
    k = decade_key(date_str)

    if k in by_key:
        p = by_key[k]
        return p["baseline_ndvi"], p.get("baseline_kind") == "history"

    if k > keys[-1]:
        last = by_key[keys[-1]]
        if len(keys) >= 2:
            prev = by_key[keys[-2]]
            step = (last["baseline_ndvi"] - prev["baseline_ndvi"]) / max(keys[-1] - keys[-2], 1)
        else:
            step = 0.0
        value = max(0.0, last["baseline_ndvi"] + step * (k - keys[-1]))
        return value, last.get("baseline_kind") == "history"

    nearest = min(keys, key=lambda key: abs(key - k))
    p = by_key[nearest]
    return p["baseline_ndvi"], p.get("baseline_kind") == "history"


def compare_seasons(current: list[dict], previous: list[dict],
                    min_common: int = 3) -> dict | None:
    """
    Сравнение текущего сезона с прошлым по совпадающим декадам.

    Сравнение одной точки (например, последней) вводило в заблуждение: при
    пропущенных из-за облаков датах сопоставлялись разные фазы сезона, а в
    конце сезона обе точки просто лежат у нуля. Здесь берётся среднее NDVI по
    декадам, для которых есть надёжные данные в обоих сезонах, — это прокси
    продуктивности посевов за период.

    Возвращает None, если совпадающих декад меньше min_common.
    """
    def by_decade(series):
        out = {}
        for p in series:
            if p.get("ndvi_mean") is None or is_low_quality(p):
                continue
            out.setdefault(decade_key(p["date"]), []).append(p["ndvi_mean"])
        return {k: mean(v) for k, v in out.items()}

    cur, prev = by_decade(current), by_decade(previous)
    common = sorted(set(cur) & set(prev))
    if len(common) < min_common:
        return None

    cur_mean = mean(cur[k] for k in common)
    prev_mean = mean(prev[k] for k in common)
    diff_pct = round(100 * (cur_mean - prev_mean) / prev_mean, 1) if prev_mean else 0.0

    return {
        "cur_mean": round(cur_mean, 3),
        "prev_mean": round(prev_mean, 3),
        "diff_pct": diff_pct,
        "n_common": len(common),
        "cur_peak": round(max(cur.values()), 3),
        "prev_peak": round(max(prev.values()), 3),
        "cur_year": current[-1]["date"][:4],
        "prev_year": previous[-1]["date"][:4],
    }
