"""
Временной ряд NDVI/NDMI по контуру поля через Sentinel Hub Statistical API.

Sentinel-2 снимает одну и ту же точку примерно раз в 5 дней в видимом и
инфракрасном диапазонах.
    NDVI = (NIR - RED) / (NIR + RED) — густота и «живость» растительности.
           Голая почва около 0.1-0.2, плотные здоровые посевы 0.6-0.9.
    NDMI = (NIR - SWIR) / (NIR + SWIR) — влагосодержание растительности.

Снимки не скачиваются: контур поля и формула расчёта (evalscript) уходят
в Statistical API, а обратно приходит уже агрегированная статистика по датам.
Растр по пикселям для того же поля получает модуль raster.py через Process API.
"""

import math
from sentinelhub import CRS, DataCollection, Geometry, SentinelHubStatistical, SHConfig

from config import SH_CLIENT_ID, SH_CLIENT_SECRET, SH_BASE_URL, SH_TOKEN_URL


def get_data_collection():
    """
    DataCollection.SENTINEL2_L2A по умолчанию привязан к сервису
    services.sentinel-hub.com. Для Copernicus Data Space Ecosystem коллекцию
    нужно пересоздать с service_url = SH_BASE_URL, иначе запрос уходит на
    старый сервис и возвращает 401 даже с корректным токеном.
    """
    return DataCollection.SENTINEL2_L2A.define_from(
        "SENTINEL2_L2A_CDSE", service_url=SH_BASE_URL
    )


def get_config() -> SHConfig:
    """Собирает конфиг авторизации для Sentinel Hub / Copernicus Data Space."""
    if not SH_CLIENT_ID or not SH_CLIENT_SECRET:
        raise RuntimeError(
            "Не заданы переменные окружения SH_CLIENT_ID / SH_CLIENT_SECRET. "
            "Установите их перед запуском реального режима (см. config.py) — "
            "для demo-режима они не нужны."
        )
    config = SHConfig()
    config.sh_client_id = SH_CLIENT_ID
    config.sh_client_secret = SH_CLIENT_SECRET
    config.sh_base_url = SH_BASE_URL
    config.sh_token_url = SH_TOKEN_URL  # отдельный identity-сервер, не sh_base_url!
    return config


# Evalscript — маленькая программа на JS, которая выполняется на стороне
# Sentinel Hub для каждого пикселя снимка. Мы просим вернуть NDVI, NDMI
# и маску облачности/валидности данных (dataMask).
EVALSCRIPT = """
//VERSION=3
function setup() {
  return {
    input: [{ bands: ["B04", "B08", "B8A", "B11", "SCL", "dataMask"] }],
    output: [
      { id: "ndvi", bands: 1 },
      { id: "ndmi", bands: 1 },
      { id: "dataMask", bands: 1 }
    ]
  };
}

function evaluatePixel(sample) {
  // SCL — Scene Classification Layer. Отсекаем облака, тени, снег.
  let invalid = [8, 9, 10, 11].includes(sample.SCL);
  let mask = invalid ? 0 : sample.dataMask;

  let ndvi = (sample.B08 - sample.B04) / (sample.B08 + sample.B04);
  let ndmi = (sample.B8A - sample.B11) / (sample.B8A + sample.B11);

  return {
    ndvi: [ndvi],
    ndmi: [ndmi],
    dataMask: [mask]
  };
}
"""


def estimate_grid_size(field_geojson: dict, target_resolution_m: int = 10,
                        min_px: int = 8, max_px: int = 250) -> tuple[int, int]:
    """
    Считает размер сетки в ПИКСЕЛЯХ (не в метрах и не в градусах!) под контур поля.

    Параметр resolution здесь не подходит: для геометрии в WGS84 Sentinel Hub
    трактует его значение как градусы, а не метры, и на всё поле приходится
    один пиксель — запрос упирается в лимит 1500 м/пиксель. Размер в пикселях
    от системы координат не зависит.

    Размеры поля в метрах оцениваются приближённо через градусы широты и
    долготы: для выбора сетки этой точности достаточно.
    """
    from math import cos, radians

    coords = field_geojson["coordinates"][0]
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]

    lon_extent_deg = max(lons) - min(lons)
    lat_extent_deg = max(lats) - min(lats)
    avg_lat = sum(lats) / len(lats)

    width_m = lon_extent_deg * 111_320 * cos(radians(avg_lat))
    height_m = lat_extent_deg * 110_540

    width_px = max(min_px, min(max_px, round(width_m / target_resolution_m)))
    height_px = max(min_px, min(max_px, round(height_m / target_resolution_m)))

    return width_px, height_px


def _safe_stat(value, digits: int = 4):
    """
    Безопасно приводит значение статистики к числу.

    Statistical API при делении на ноль в evalscript (например, B08+B04=0
    на кромке облака) присылает "mean": "NaN" строкой, а не числом, и
    round() на такой строке падает. float() распознаёт и число, и "NaN",
    после чего точка отбрасывается как невалидная.
    """
    if value is None:
        return None
    try:
        fval = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(fval) or math.isinf(fval):
        return None
    return round(fval, digits)


def fetch_ndvi_ndmi_timeseries(field_geojson: dict, date_from: str, date_to: str,
                                aggregation_days: int = 10, max_cloud_coverage: float = 0.6):
    """
    Получает временной ряд NDVI/NDMI по контуру поля.

    Параметры:
        field_geojson: контур поля в формате GeoJSON Polygon (координаты WGS84)
        date_from, date_to: границы периода "YYYY-MM-DD"
        aggregation_days: шаг агрегации в днях (10 = по декадам, как в задаче ТЗ)
        max_cloud_coverage: отсекать снимки с облачностью выше этого порога

    Возвращает:
        список словарей вида:
        {"date": "2026-05-01", "ndvi_mean": 0.62, "ndmi_mean": 0.31, "valid_pixels": 843}
    """
    config = get_config()
    geometry = Geometry(field_geojson, crs=CRS.WGS84)

    size = estimate_grid_size(field_geojson)

    aggregation = SentinelHubStatistical.aggregation(
        evalscript=EVALSCRIPT,
        time_interval=(date_from, date_to),
        aggregation_interval=f"P{aggregation_days}D",
        size=size,  # (ширина, высота) в пикселях — не зависит от CRS геометрии
    )

    input_data = SentinelHubStatistical.input_data(
        get_data_collection(),
        maxcc=max_cloud_coverage,
    )

    request = SentinelHubStatistical(
        aggregation=aggregation,
        input_data=[input_data],
        geometry=geometry,
        config=config,
    )

    raw = request.get_data()[0]

    results = []
    for interval in raw.get("data", []):
        date = interval["interval"]["from"][:10]
        outputs = interval.get("outputs", {})

        ndvi_stats = outputs.get("ndvi", {}).get("bands", {}).get("B0", {}).get("stats")
        ndmi_stats = outputs.get("ndmi", {}).get("bands", {}).get("B0", {}).get("stats")

        if not ndvi_stats:
            continue

        # sampleCount — все пиксели сетки внутри контура, включая закрытые
        # облаком; noDataCount — те из них, что отсечены маской. Валидные
        # пиксели — разность.
        total_pixels = int(ndvi_stats.get("sampleCount", 0) or 0)
        no_data = int(ndvi_stats.get("noDataCount", 0) or 0)
        valid_pixels = total_pixels - no_data
        if valid_pixels <= 0:
            continue  # весь контур закрыт облаком — данных за период нет

        ndvi_mean = _safe_stat(ndvi_stats.get("mean"))
        if ndvi_mean is None:
            continue  # NDVI не удалось посчитать (NaN/деление на ноль) — пропускаем точку

        results.append({
            "date": date,
            "ndvi_mean": ndvi_mean,
            "ndvi_std": _safe_stat(ndvi_stats.get("stDev"), 4) or 0,
            "ndmi_mean": _safe_stat(ndmi_stats.get("mean")) if ndmi_stats else None,
            "ndmi_std": _safe_stat(ndmi_stats.get("stDev"), 4) if ndmi_stats else None,
            "valid_pixels": valid_pixels,
            "total_pixels": total_pixels,
        })

    return results
