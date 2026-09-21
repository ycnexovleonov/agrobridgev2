"""
Сохранённый снимок реальных данных Sentinel-2.

Зачем: реальный режим требует OAuth-ключей Copernicus. Чтобы приложение можно
было запустить и проверить без ключей и без интернета, настоящие данные один
раз сохраняются в репозиторий (`python make_snapshot.py`), а приложение
читает их из файлов. Это те же значения, которые вернул Sentinel Hub, а не
синтетика, поэтому режим помечается как реальные данные с датой получения.

Состав снимка (папка data/snapshot/):
    meta.json     — контур поля, границы сезона, дата получения, список дат растров
    series.json   — временной ряд текущего сезона и прошлых сезонов
    rasters.npz   — растры NDVI и маски по датам (сжатый npz)
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SNAPSHOT_DIR = Path(__file__).resolve().parent / "data" / "snapshot"
_META = "meta.json"
_SERIES = "series.json"
_RASTERS = "rasters.npz"


def snapshot_available(directory: Path = SNAPSHOT_DIR) -> bool:
    return all((directory / f).exists() for f in (_META, _SERIES))


def save_snapshot(field_geojson: dict, date_from: str, date_to: str,
                  timeseries: list[dict], history: list[list[dict]],
                  rasters: dict[str, tuple[np.ndarray, np.ndarray]],
                  directory: Path = SNAPSHOT_DIR) -> None:
    """Записывает снимок на диск. rasters: {дата: (ndvi float32, mask)}."""
    directory.mkdir(parents=True, exist_ok=True)
    meta = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "source": "Sentinel-2 L2A, Copernicus Data Space Ecosystem (Sentinel Hub API)",
        "field_geojson": field_geojson,
        "date_from": date_from,
        "date_to": date_to,
        "raster_dates": sorted(rasters),
    }
    (directory / _META).write_text(json.dumps(meta, ensure_ascii=False, indent=2), "utf-8")
    (directory / _SERIES).write_text(
        json.dumps({"current": timeseries, "history": history}, ensure_ascii=False), "utf-8")

    arrays = {}
    for date, (ndvi, mask) in rasters.items():
        arrays[f"ndvi_{date}"] = np.asarray(ndvi, dtype=np.float32)
        arrays[f"mask_{date}"] = (np.asarray(mask) > 0.5).astype(np.uint8)
    if arrays:
        np.savez_compressed(directory / _RASTERS, **arrays)


def load_snapshot(directory: Path = SNAPSHOT_DIR) -> dict | None:
    """Возвращает {meta, timeseries, history} или None, если снимка нет или он повреждён."""
    if not snapshot_available(directory):
        return None
    try:
        meta = json.loads((directory / _META).read_text("utf-8"))
        series = json.loads((directory / _SERIES).read_text("utf-8"))
        return {"meta": meta, "timeseries": series["current"], "history": series["history"]}
    except (OSError, ValueError, KeyError):
        return None


def load_snapshot_raster(date: str, directory: Path = SNAPSHOT_DIR):
    """(ndvi, mask) на дату или None, если для даты растр не сохранён."""
    path = directory / _RASTERS
    if not path.exists():
        return None
    with np.load(path) as data:
        if f"ndvi_{date}" not in data.files:
            return None
        return (data[f"ndvi_{date}"].astype(np.float32),
                data[f"mask_{date}"].astype(np.float32))
