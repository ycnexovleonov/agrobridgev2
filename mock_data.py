"""
Генератор синтетического временного ряда NDVI/NDMI для demo-режима.

Нужен, чтобы разрабатывать и показывать интерфейс без обращения к API:
без интернета, без расхода лимитов аккаунта и без задержек сервиса.
В интерфейсе режим явно подписан как демонстрационный.
"""

import random
from datetime import datetime, timedelta

TOTAL_PIXELS = 900


def generate_mock_timeseries(date_from: str, date_to: str, aggregation_days: int = 10,
                              with_anomaly: bool = True, seed: int = 42):
    """Генерирует реалистичную сезонную кривую NDVI/NDMI с ростом и спадом."""
    rng = random.Random(seed)

    start = datetime.strptime(date_from, "%Y-%m-%d")
    end = datetime.strptime(date_to, "%Y-%m-%d")

    dates = []
    current = start
    while current <= end:
        dates.append(current)
        current += timedelta(days=aggregation_days)

    n = len(dates)
    results = []

    # Отдельный генератор для покрытия снимком: не сдвигает основную кривую.
    # Две точки ряда имитируют сильную облачность.
    quality_rng = random.Random(seed + 1000)
    low_quality_idx = {3, 11} if n > 12 else set()
    for i, date in enumerate(dates):
        # Сезонная кривая: рост -> пик -> спад (типичная динамика вегетации)
        progress = i / max(n - 1, 1)
        seasonal_curve = 0.15 + 0.65 * (1 - abs(2 * progress - 1) ** 1.5)

        noise = rng.uniform(-0.03, 0.03)
        ndvi = round(max(0.05, min(0.9, seasonal_curve + noise)), 4)
        ndmi = round(max(0.0, min(0.6, ndvi * 0.5 + rng.uniform(-0.02, 0.02))), 4)

        # Искусственная аномалия в середине сезона — имитирует засуху/вредителя
        if with_anomaly and 0.35 < progress < 0.55:
            ndvi = round(ndvi * 0.55, 4)
            ndmi = round(ndmi * 0.6, 4)

        valid_pixels = int(TOTAL_PIXELS * quality_rng.uniform(0.7, 1.0))
        if i in low_quality_idx:
            valid_pixels = int(TOTAL_PIXELS * quality_rng.uniform(0.08, 0.25))

        results.append({
            "date": date.strftime("%Y-%m-%d"),
            "ndvi_mean": ndvi,
            "ndvi_std": round(rng.uniform(0.02, 0.08), 4),
            "ndmi_mean": ndmi,
            "ndmi_std": round(rng.uniform(0.02, 0.05), 4),
            "valid_pixels": valid_pixels,
            "total_pixels": TOTAL_PIXELS,
        })

    return results
