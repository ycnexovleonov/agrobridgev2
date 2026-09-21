"""
raster.py

Растровый слой NDVI: значение индекса для каждого пикселя внутри контура поля,
а не одно усреднённое число по всему полю.

Отличие от sentinel_client.py:
    Statistical API -> агрегированная статистика по датам (одно число на поле)
    Process API     -> массив значений NDVI по пикселям (картинка поля)

Пайплайн:
    1. Process API возвращает GeoTIFF из двух float32-каналов: NDVI и маска
       валидности (0 там, где облако/тень/снег или пиксель вне контура поля).
    2. Массив NDVI раскрашивается палитрой и превращается в PNG с прозрачным фоном.
    3. PNG накладывается на карту Folium через ImageOverlay по bbox контура.
    4. Параллельно по тому же массиву считается доля площади поля ниже порога —
       это и есть количественная оценка проблемных зон.
"""

import base64
import io
import math

import numpy as np

from config import ANOMALY_THRESHOLD, NON_CROP_NDVI, ZONE_SIGMA

# sentinelhub импортируется внутри fetch_ndvi_raster, а не здесь: demo-режим
# и вся обработка массивов должны работать даже без установленной библиотеки.

# Значение NDVI, которому соответствует нижний и верхний край цветовой шкалы.
# 0.15 — голая/убранная почва, 0.75 — плотная здоровая вегетация.
NDVI_COLOR_MIN = 0.15
NDVI_COLOR_MAX = 0.75

RASTER_EVALSCRIPT = """
//VERSION=3
function setup() {
  return {
    input: [{ bands: ["B04", "B08", "SCL", "dataMask"] }],
    output: { bands: 2, sampleType: "FLOAT32" }
  };
}

function evaluatePixel(sample) {
  // SCL: 0 no data, 1 saturated, 3 cloud shadow, 8/9 cloud, 10 cirrus, 11 snow
  var bad = [0, 1, 3, 8, 9, 10, 11].indexOf(sample.SCL) > -1;
  var mask = (bad || sample.dataMask === 0) ? 0 : 1;
  var ndvi = index(sample.B08, sample.B04);
  return [ndvi, mask];
}
"""


def field_bbox(field_geojson: dict) -> tuple[float, float, float, float]:
    """Габаритный прямоугольник контура: (min_lon, min_lat, max_lon, max_lat)."""
    coords = field_geojson["coordinates"][0]
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    return min(lons), min(lats), max(lons), max(lats)


def field_area_ha(field_geojson: dict) -> float:
    """
    Площадь контура в гектарах по формуле шнурков.

    Координаты в градусах переводятся в метры локально: долгота сжимается
    множителем cos(широта), поэтому пересчёт делается относительно средней
    широты контура. Для одного поля погрешность такого приближения
    пренебрежимо мала.
    """
    coords = field_geojson["coordinates"][0]
    lats = [c[1] for c in coords]
    avg_lat = sum(lats) / len(lats)

    xs = [c[0] * 111_320 * math.cos(math.radians(avg_lat)) for c in coords]
    ys = [c[1] * 110_540 for c in coords]

    area_m2 = 0.0
    for i in range(len(coords) - 1):
        area_m2 += xs[i] * ys[i + 1] - xs[i + 1] * ys[i]
    return round(abs(area_m2) / 2 / 10_000, 1)


def raster_grid_size(field_geojson: dict, target_resolution_m: int = 10,
                     min_px: int = 32, max_px: int = 512) -> tuple[int, int]:
    """
    Размер растра в пикселях. Берётся крупнее, чем для Statistical API:
    здесь картинку смотрят глазами, и сетка 8x8 была бы нечитаемой.
    Верхний предел 512 удерживает объём ответа и время запроса в разумных рамках.
    """
    min_lon, min_lat, max_lon, max_lat = field_bbox(field_geojson)
    avg_lat = (min_lat + max_lat) / 2

    width_m = (max_lon - min_lon) * 111_320 * math.cos(math.radians(avg_lat))
    height_m = (max_lat - min_lat) * 110_540

    width_px = max(min_px, min(max_px, round(width_m / target_resolution_m)))
    height_px = max(min_px, min(max_px, round(height_m / target_resolution_m)))
    return width_px, height_px


def fetch_ndvi_raster(field_geojson: dict, date_from: str, date_to: str,
                      max_cloud_coverage: float = 0.6):
    """
    Запрашивает растр NDVI по контуру поля за период.

    За период берётся один снимок — наименее облачный (mosaicking_order=leastCC),
    иначе Process API усреднит несколько дат и размоет реальную картину.
    Период поэтому задаётся узким окном вокруг интересующей даты.

    Возвращает: (ndvi_array, mask_array) — два float32-массива формы (H, W).
    """
    from sentinelhub import (CRS, BBox, Geometry, MimeType, MosaickingOrder,
                             SentinelHubRequest)

    from sentinel_client import get_config, get_data_collection

    config = get_config()
    bbox = BBox(bbox=field_bbox(field_geojson), crs=CRS.WGS84)
    geometry = Geometry(field_geojson, crs=CRS.WGS84)
    size = raster_grid_size(field_geojson)

    request = SentinelHubRequest(
        evalscript=RASTER_EVALSCRIPT,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=get_data_collection(),
                time_interval=(date_from, date_to),
                mosaicking_order=MosaickingOrder.LEAST_CC,
                maxcc=max_cloud_coverage,
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox,
        geometry=geometry,
        size=size,
        config=config,
    )

    data = request.get_data()[0]
    array = np.asarray(data, dtype=np.float32)

    if array.ndim == 2:  # сервис вернул один канал — маску считаем сплошной
        return array, np.ones_like(array)
    return array[..., 0], array[..., 1]


def _colormap(name: str = "RdYlGn"):
    """Палитра matplotlib: красный — низкий NDVI, зелёный — высокий."""
    try:
        import matplotlib
        return matplotlib.colormaps[name]
    except (AttributeError, KeyError):
        from matplotlib import cm
        return cm.get_cmap(name)


def ndvi_raster_to_png(ndvi: np.ndarray, mask: np.ndarray,
                       vmin: float = NDVI_COLOR_MIN, vmax: float = NDVI_COLOR_MAX,
                       opacity: int = 205) -> bytes:
    """
    Раскрашивает массив NDVI и возвращает PNG с альфа-каналом.

    Пиксели с mask == 0 (облако, тень, за границей поля) делаются полностью
    прозрачными — на карте под ними видна подложка, а не ложный цвет.
    """
    from PIL import Image

    valid = mask > 0.5
    scaled = np.clip((ndvi - vmin) / (vmax - vmin), 0.0, 1.0)
    scaled = np.nan_to_num(scaled, nan=0.0)

    rgba = (_colormap()(scaled) * 255).astype(np.uint8)
    rgba[..., 3] = np.where(valid, opacity, 0).astype(np.uint8)

    buffer = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buffer, format="PNG")
    return buffer.getvalue()


def png_to_data_uri(png_bytes: bytes) -> str:
    """PNG в base64 data-URI — Folium принимает такую строку как источник ImageOverlay."""
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")


def crop_pixel_mask(ndvi: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Валидные пиксели, похожие на посевы: без облака, за контуром и без воды,
    глубокой тени и построек (NDVI ниже NON_CROP_NDVI)."""
    return (mask > 0.5) & (ndvi >= NON_CROP_NDVI)


def problem_pixel_mask(ndvi: np.ndarray, mask: np.ndarray, threshold: float) -> np.ndarray:
    """Единое определение проблемного пикселя для статистики, слоя и контуров."""
    return crop_pixel_mask(ndvi, mask) & (ndvi < threshold)


def zone_statistics(ndvi: np.ndarray, mask: np.ndarray,
                    threshold_ratio: float = ANOMALY_THRESHOLD,
                    field_area: float | None = None) -> dict:
    """
    Количественная оценка проблемных зон внутри поля.

    Порог считается от МЕДИАНЫ NDVI посевных пикселей этого же снимка.
    Пиксель проблемный, если он одновременно ниже медианы на |ANOMALY_THRESHOLD|
    и ниже её больше чем на ZONE_SIGMA робастных сигм (1.4826 * MAD). Второе
    условие не даёт принять за проблему обычный разброс значений на однородном
    поле, первое — отсекает статистически значимые, но агрономически
    ничтожные отклонения на очень ровном. Сравнение идёт с самим полем в тот же
    день, поэтому оценка не зависит от стадии сезона.

    Вода, глубокая тень и постройки (NDVI < NON_CROP_NDVI) в расчёт типичного
    уровня не входят и не могут быть проблемной зоной; их доля показывается
    отдельно. Доля проблемной площади считается от всех валидных пикселей
    контура, поэтому непосевные участки не завышают её.
    """
    valid = mask > 0.5
    crop = crop_pixel_mask(ndvi, mask)
    valid_count = int(valid.sum())
    crop_values = ndvi[crop]

    if valid_count == 0 or crop_values.size == 0:
        return {"error": "Нет валидных пикселей посевов — снимок закрыт облаками "
                         "или контур не содержит растительности."}

    median = float(np.nanmedian(crop_values))
    # робастная сигма: 1.4826 * MAD, не зависит от редких тёмных выбросов
    sigma = 1.4826 * float(np.nanmedian(np.abs(crop_values - median)))
    threshold = min(median * (1 + threshold_ratio), median - ZONE_SIGMA * sigma)
    problem = int(np.count_nonzero(crop & (ndvi < threshold)))
    problem_share = problem / valid_count

    stats = {
        "field_mean_ndvi": round(float(np.nanmean(crop_values)), 4),
        "field_median_ndvi": round(median, 4),
        "threshold_ndvi": round(threshold, 4),
        "robust_sigma": round(sigma, 4),
        "valid_pixels": valid_count,
        "total_pixels": int(mask.size),
        "coverage_pct": round(100 * valid_count / mask.size, 1),
        "non_crop_share_pct": round(100 * (valid_count - crop_values.size) / valid_count, 1),
        "problem_pixels": problem,
        "problem_share_pct": round(100 * problem_share, 1),
        "min_ndvi": round(float(np.nanmin(crop_values)), 4),
        "max_ndvi": round(float(np.nanmax(crop_values)), 4),
    }

    if field_area:
        stats["problem_area_ha"] = round(field_area * problem_share, 1)
        stats["field_area_ha"] = field_area

    return stats


def problem_zone_mask_png(ndvi: np.ndarray, mask: np.ndarray,
                          threshold: float, opacity: int = 190) -> bytes:
    """
    Бинарный слой: только проблемные пиксели, залитые красным, остальное прозрачно.
    Отдельно от цветовой шкалы NDVI — на нём сразу видно, где именно выезжать в поле.
    """
    from PIL import Image

    problem = problem_pixel_mask(ndvi, mask, threshold)

    rgba = np.zeros((*ndvi.shape, 4), dtype=np.uint8)
    rgba[..., 0] = 214
    rgba[..., 1] = 39
    rgba[..., 2] = 40
    rgba[..., 3] = np.where(problem, opacity, 0).astype(np.uint8)

    buffer = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buffer, format="PNG")
    return buffer.getvalue()


def folium_bounds(field_geojson: dict) -> list[list[float]]:
    """Границы для ImageOverlay в порядке Folium: [[юг, запад], [север, восток]]."""
    min_lon, min_lat, max_lon, max_lat = field_bbox(field_geojson)
    return [[min_lat, min_lon], [max_lat, max_lon]]


def mock_ndvi_raster(field_geojson: dict, seed: int = 42, base: float = 0.62,
                     strength: float = 0.33) -> tuple[np.ndarray, np.ndarray]:
    """
    Синтетический растр для demo-режима.

    base     — средний уровень NDVI поля на выбранную дату (берётся из ряда),
    strength — глубина просевшего пятна: на дате аномалии оно выраженное,
               на здоровой дате почти незаметное.

    В растре есть просевшее пятно в юго-западной части поля и участок,
    закрытый облаком, в северо-восточном углу. Нужен, чтобы интерфейс и логика
    зон проверялись без обращения к API.
    """
    width, height = raster_grid_size(field_geojson)
    rng = np.random.default_rng(seed)

    yy, xx = np.mgrid[0:height, 0:width]
    ndvi = base + 0.05 * np.sin(xx / 18.0) + 0.04 * np.cos(yy / 14.0)
    ndvi += rng.normal(0, 0.02, size=(height, width))

    cy, cx = height * 0.70, width * 0.28
    dist = ((yy - cy) / (height * 0.18)) ** 2 + ((xx - cx) / (width * 0.16)) ** 2
    ndvi -= strength * np.exp(-dist)

    mask = np.ones((height, width), dtype=np.float32)
    mask[: int(height * 0.14), int(width * 0.82):] = 0.0

    return np.clip(ndvi, -0.1, 0.95).astype(np.float32), mask


# ---------------------------------------------------------------------------
# Легенда и векторизация проблемных зон
# ---------------------------------------------------------------------------

MIN_ZONE_PIXELS = 30   # зоны мельче (около 0.3 га при 10 м/пиксель) считаются шумом
MAX_ZONES = 30         # верхний предел числа зон в отчёте


def legend_colors(n: int = 7) -> list[str]:
    """Цвета шкалы NDVI в hex — для легенды на карте. Совпадают с палитрой растра."""
    from matplotlib.colors import to_hex
    cmap = _colormap()
    return [to_hex(cmap(i / (n - 1))) for i in range(n)]


def _signed_area(ring: np.ndarray) -> float:
    """Ориентированная площадь кольца по формуле шнурков (в единицах координат)."""
    x, y = ring[:, 0], ring[:, 1]
    return float(0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))


def _component_rings(component_mask: np.ndarray) -> list[np.ndarray]:
    """
    Контуры одной связной области в координатах пиксельной сетки (x вправо,
    y вниз, начало в левом верхнем углу растра).

    Контур строится маршевыми квадратами (contourpy, он идёт вместе с
    matplotlib) на уровне 0.5 по бинарной маске с нулевой рамкой: рамка
    гарантирует, что области на краю растра тоже дают замкнутые кольца.
    """
    from contourpy import contour_generator

    padded = np.pad(component_mask.astype(np.float64), 1)
    lines = contour_generator(z=padded).lines(0.5)

    rings = []
    for line in lines:
        pts = np.asarray(line, dtype=np.float64)
        if len(pts) < 4:
            continue
        # -1 снимает рамку; +0.5 переводит из индекса центра пикселя в
        # координату края пикселя, чтобы 0 соответствовал левому краю растра
        pts = pts - 1.0 + 0.5
        if not np.allclose(pts[0], pts[-1]):
            pts = np.vstack([pts, pts[0]])
        rings.append(pts)
    return rings



def _simplify_ring(ring: np.ndarray, tolerance_px: float = 0.6) -> np.ndarray:
    """
    Упрощение контура алгоритмом Дугласа-Пойкера (в координатах пикселей).

    Контур маршевых квадратов повторяет пиксельную «лесенку», и его вершины
    почти избыточны. Допуск 0.6 пикселя (около 6 м) убирает лишние вершины,
    не сдвигая границу заметно на карте. Кольцо остаётся замкнутым.
    """
    if len(ring) <= 8:
        return ring

    open_ring = ring[:-1]
    keep = np.zeros(len(open_ring), dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(open_ring) - 1)]

    while stack:
        start, end = stack.pop()
        if end <= start + 1:
            continue
        a, b = open_ring[start], open_ring[end]
        segment = b - a
        length = np.hypot(*segment)
        pts = open_ring[start + 1:end]
        if length == 0:
            distances = np.hypot(*(pts - a).T)
        else:
            distances = np.abs(segment[0] * (pts[:, 1] - a[1]) -
                               segment[1] * (pts[:, 0] - a[0])) / length
        far = int(np.argmax(distances))
        if distances[far] > tolerance_px:
            mid = start + 1 + far
            keep[mid] = True
            stack.append((start, mid))
            stack.append((mid, end))

    simplified = open_ring[keep]
    if len(simplified) < 3:
        return ring
    return np.vstack([simplified, simplified[0]])

def _orient(ring: np.ndarray, counter_clockwise: bool) -> np.ndarray:
    """Приводит кольцо к нужному направлению обхода (RFC 7946: внешнее против часовой)."""
    # ось y растра направлена вниз, поэтому после перехода к широте направление
    # инвертируется — ориентация проверяется уже в географических координатах
    area = _signed_area(ring)
    if (area > 0) != counter_clockwise:
        return ring[::-1]
    return ring


def extract_problem_zones(ndvi: np.ndarray, mask: np.ndarray, threshold: float,
                          field_geojson: dict, field_area: float | None = None,
                          min_pixels: int = MIN_ZONE_PIXELS,
                          max_zones: int = MAX_ZONES) -> list[dict]:
    """
    Выделяет проблемные зоны как векторные полигоны (GeoJSON Features).

    Пиксели с NDVI ниже порога объединяются в связные области (по сторонам,
    без диагоналей), каждая оформляется полигоном с отверстиями, если внутри
    зоны есть нормальные участки. Мелкие области отбрасываются как шум.

    Площадь зоны — доля её пикселей среди валидных, умноженная на площадь
    контура. Так сумма по зонам совпадает с общей площадью проблемных зон
    в статистике, и облачные участки не занижают результат.
    """
    from matplotlib.path import Path
    from scipy import ndimage

    valid_count = int((mask > 0.5).sum())
    problem = problem_pixel_mask(ndvi, mask, threshold)
    if valid_count == 0 or not problem.any():
        return []

    height, width = ndvi.shape
    min_lon, min_lat, max_lon, max_lat = field_bbox(field_geojson)
    lon_per_px = (max_lon - min_lon) / width
    lat_per_px = (max_lat - min_lat) / height

    def to_lonlat(pts: np.ndarray) -> np.ndarray:
        lon = min_lon + pts[:, 0] * lon_per_px
        lat = max_lat - pts[:, 1] * lat_per_px
        return np.column_stack([lon, lat])

    labels, count = ndimage.label(problem)
    sizes = ndimage.sum(problem, labels, index=range(1, count + 1))

    order = [i for i in np.argsort(sizes)[::-1] if sizes[i] >= min_pixels][:max_zones]

    features = []
    for rank, idx in enumerate(order, start=1):
        label_id = idx + 1
        comp = labels == label_id
        pixel_count = int(sizes[idx])

        rings = [to_lonlat(_simplify_ring(r)) for r in _component_rings(comp)]
        if not rings:
            continue

        rings.sort(key=lambda r: abs(_signed_area(r)), reverse=True)
        exterior = rings[0]
        exterior_path = Path(exterior)

        holes, extra = [], []
        for ring in rings[1:]:
            if exterior_path.contains_point(ring[0]):
                holes.append(ring)
            else:
                extra.append(ring)

        polygon = [_orient(exterior, True).tolist()] + [_orient(h, False).tolist() for h in holes]

        ys, xs = np.nonzero(comp)
        centroid_lon = min_lon + (xs.mean() + 0.5) * lon_per_px
        centroid_lat = max_lat - (ys.mean() + 0.5) * lat_per_px

        zone_values = ndvi[comp]
        share = pixel_count / valid_count

        props = {
            "kind": "problem_zone",
            "zone_id": rank,
            "pixels": pixel_count,
            "share_of_field_pct": round(100 * share, 2),
            "mean_ndvi": round(float(np.nanmean(zone_values)), 4),
            "min_ndvi": round(float(np.nanmin(zone_values)), 4),
            "centroid_lat": round(float(centroid_lat), 6),
            "centroid_lon": round(float(centroid_lon), 6),
        }
        if field_area:
            props["area_ha"] = round(field_area * share, 2)

        features.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": polygon},
            "properties": props,
        })

    return features



# ---------------------------------------------------------------------------
# Разбор загруженного контура поля
# ---------------------------------------------------------------------------

def _ring_area(ring: list) -> float:
    """Модуль площади кольца в градусах² — нужен только чтобы выбрать самый большой полигон."""
    total = 0.0
    for i in range(len(ring) - 1):
        total += ring[i][0] * ring[i + 1][1] - ring[i + 1][0] * ring[i][1]
    return abs(total) / 2


def normalize_field_geometry(obj: dict) -> dict:
    """
    Приводит загруженный GeoJSON к простому Polygon в координатах [lon, lat].

    Принимает FeatureCollection, Feature, Polygon и MultiPolygon (берётся самый
    большой полигон); высоту (третью координату) отбрасывает; внутренние
    кольца отбрасывает — контур поля нужен внешний. Некорректный файл даёт
    ValueError с понятным сообщением, а не падение внутри библиотек.
    """
    if not isinstance(obj, dict):
        raise ValueError("файл не содержит объект GeoJSON")

    kind = obj.get("type")
    if kind == "FeatureCollection":
        geoms = [f.get("geometry") for f in obj.get("features", []) if f.get("geometry")]
        if not geoms:
            raise ValueError("в FeatureCollection нет геометрии")
        candidates = [normalize_field_geometry(g) for g in geoms
                      if g.get("type") in ("Polygon", "MultiPolygon")]
        if not candidates:
            raise ValueError("в файле нет полигонов (найдены только точки или линии)")
        return max(candidates, key=lambda g: _ring_area(g["coordinates"][0]))
    if kind == "Feature":
        if not obj.get("geometry"):
            raise ValueError("у объекта Feature нет геометрии")
        return normalize_field_geometry(obj["geometry"])

    if kind == "Polygon":
        rings = obj.get("coordinates")
    elif kind == "MultiPolygon":
        polygons = obj.get("coordinates") or []
        if not polygons:
            raise ValueError("MultiPolygon пуст")
        rings = max(polygons, key=lambda p: _ring_area(p[0]))
    else:
        raise ValueError(f"тип геометрии «{kind}» не поддерживается, нужен Polygon")

    if not rings or not rings[0]:
        raise ValueError("у полигона нет внешнего контура")

    ring = [[float(c[0]), float(c[1])] for c in rings[0]]
    if ring[0] != ring[-1]:
        ring.append(ring[0])
    if len(ring) < 4:
        raise ValueError("в контуре меньше трёх различных вершин")
    if not all(-180 <= x <= 180 and -90 <= y <= 90 for x, y in ring):
        raise ValueError("координаты вне диапазона: ожидается порядок [долгота, широта]")

    return {"type": "Polygon", "coordinates": [ring]}
