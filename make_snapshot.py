"""
Сохраняет снимок реальных данных Sentinel-2 в data/snapshot/.

Запускается один раз, с ключами Copernicus (SH_CLIENT_ID / SH_CLIENT_SECRET) и
доступом в интернет. После этого приложение работает на настоящих данных без
ключей: режим «Реальные данные (сохранённый снимок)».

    python make_snapshot.py
    python make_snapshot.py --field my_field.geojson --from 2026-04-01 --to 2026-09-15

Что сохраняется:
    - временной ряд NDVI/NDMI текущего сезона;
    - те же периоды за HISTORY_YEARS прошлых сезонов (для нормы);
    - растр NDVI для каждой даты с достаточным покрытием снимком
      (не больше --max-rasters дат, самые полные по покрытию).

Запросы к API тратят квоту аккаунта: 1 запрос на сезон плюс 1 на каждый растр.
"""

import argparse
import json
from datetime import datetime, timedelta

from analysis import data_share, is_low_quality, shift_year
from config import DEFAULT_FIELD_GEOJSON, HISTORY_YEARS, SEASON_END, SEASON_START
from data_cache import SNAPSHOT_DIR, save_snapshot
from raster import fetch_ndvi_raster, normalize_field_geometry
from sentinel_client import fetch_ndvi_ndmi_timeseries


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--field", help="GeoJSON с контуром поля (по умолчанию — из config.py)")
    parser.add_argument("--from", dest="date_from", default=SEASON_START)
    parser.add_argument("--to", dest="date_to", default=SEASON_END)
    parser.add_argument("--max-rasters", type=int, default=12,
                        help="сколько растров сохранить (по умолчанию 12)")
    args = parser.parse_args()

    if args.field:
        with open(args.field, encoding="utf-8") as fh:
            field = normalize_field_geometry(json.load(fh))
    else:
        field = DEFAULT_FIELD_GEOJSON

    print(f"1) Ряд текущего сезона {args.date_from} — {args.date_to}...")
    current = fetch_ndvi_ndmi_timeseries(field, args.date_from, args.date_to)
    if not current:
        raise SystemExit("Sentinel Hub не вернул данных за текущий сезон — снимок не сохранён.")
    print(f"   получено периодов: {len(current)}")

    history = []
    for k in range(1, HISTORY_YEARS + 1):
        d_from, d_to = shift_year(args.date_from, -k), shift_year(args.date_to, -k)
        print(f"2) История: {d_from} — {d_to}...")
        try:
            data = fetch_ndvi_ndmi_timeseries(field, d_from, d_to)
        except Exception as exc:  # один неудачный год не должен ронять весь снимок
            print(f"   пропущен: {type(exc).__name__}: {exc}")
            continue
        if data:
            history.append(data)
            print(f"   получено периодов: {len(data)}")

    reliable = [p for p in current if not is_low_quality(p)]
    reliable.sort(key=lambda p: data_share(p) or 0, reverse=True)
    chosen = sorted(p["date"] for p in reliable[:args.max_rasters])

    rasters = {}
    for date in chosen:
        center = datetime.strptime(date, "%Y-%m-%d")
        w_from = (center - timedelta(days=5)).strftime("%Y-%m-%d")
        w_to = (center + timedelta(days=5)).strftime("%Y-%m-%d")
        print(f"3) Растр {date} (окно {w_from} — {w_to})...")
        try:
            rasters[date] = fetch_ndvi_raster(field, w_from, w_to)
        except Exception as exc:
            print(f"   пропущен: {type(exc).__name__}: {exc}")

    save_snapshot(field, args.date_from, args.date_to, current, history, rasters)
    print(f"\nГотово: {SNAPSHOT_DIR}")
    print(f"   сезонов истории: {len(history)}, растров: {len(rasters)}")
    print("   Добавьте папку data/snapshot в репозиторий (git add data/snapshot).")


if __name__ == "__main__":
    main()
