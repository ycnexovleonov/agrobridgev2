"""
Проверка доступа к Sentinel Hub без запуска интерфейса.

Запускать до streamlit run app.py: ошибки credentials, контура поля или
доступа к API видны здесь в чистом виде, а не скрыты за обработчиками
исключений внутри Streamlit.

    python test_connection.py
"""

from config import DEFAULT_FIELD_GEOJSON, SEASON_START, SEASON_END

print("1) Проверяю импорт и авторизацию...")
try:
    from sentinel_client import get_config, fetch_ndvi_ndmi_timeseries
    config = get_config()
    print(f"   Client ID: {config.sh_client_id[:8]}... (OK, загружен)")
except Exception as e:
    print(f"   ОШИБКА на этапе импорта/конфига: {e}")
    raise SystemExit(1)

print("\n2) Отправляю тестовый запрос к Sentinel Hub Statistical API...")
print("   Поле: контур по умолчанию (Зерендинский район, трасса А-13)")
print(f"   Период: {SEASON_START} — {SEASON_END}")

try:
    result = fetch_ndvi_ndmi_timeseries(
        DEFAULT_FIELD_GEOJSON,
        SEASON_START,
        SEASON_END,
        aggregation_days=10,
    )
    print(f"\n✅ Успех! Получено {len(result)} периодов наблюдения.\n")
    for point in result[:5]:
        share = round(100 * point['valid_pixels'] / point['total_pixels']) if point.get('total_pixels') else '?'
        print(f"   {point['date']}: NDVI={point['ndvi_mean']}, NDMI={point['ndmi_mean']}, "
              f"валидных пикселей {point['valid_pixels']} из {point['total_pixels']} ({share}%)")
    if len(result) > 5:
        print(f"   ... и ещё {len(result) - 5} периодов")

except Exception as e:
    import traceback
    print(f"\n❌ ОШИБКА при запросе к API:")
    print(f"   {type(e).__name__}: {e}")
    print("\n--- Полный traceback (для диагностики) ---")
    traceback.print_exc()
    print("--- конец traceback ---\n")
    print("   Частые причины:")
    print("   - Неверный client_id/client_secret (проверьте config.py)")
    print("   - Истёк срок жизни OAuth-клиента (пересоздайте в Dashboard)")
    print("   - Слишком строгий max_cloud_coverage — за весь период всё в облаках")
    print("   - Контур поля некорректен (проверьте порядок координат: [lon, lat])")
    raise SystemExit(1)

print("\n   Если «валидных пикселей» заметно меньше общего числа на части дат — маска "
      "облаков работает. Если везде 100% даже в апреле, напишите мне.")

print("\n3) Проверяю растровый слой (Process API)...")
try:
    from raster import (fetch_ndvi_raster, field_area_ha, ndvi_raster_to_png,
                        zone_statistics)

    last_date = result[-1]["date"]
    from datetime import datetime, timedelta
    center = datetime.strptime(last_date, "%Y-%m-%d")
    window_from = (center - timedelta(days=5)).strftime("%Y-%m-%d")
    window_to = (center + timedelta(days=5)).strftime("%Y-%m-%d")

    print(f"   Снимок за окно {window_from} — {window_to}")
    ndvi_array, mask_array = fetch_ndvi_raster(DEFAULT_FIELD_GEOJSON, window_from, window_to)
    area = field_area_ha(DEFAULT_FIELD_GEOJSON)
    stats = zone_statistics(ndvi_array, mask_array, field_area=area)

    print(f"\n   Растр получен: сетка {ndvi_array.shape[1]}x{ndvi_array.shape[0]} пикселей")
    print(f"   Площадь поля: {area} га")
    for key, value in stats.items():
        print(f"   {key}: {value}")

    # Исходный массив: цветная картинка обрезана шкалой на 0.15, и по ней нельзя
    # понять, какие значения у тёмных пикселей (вода, тени, стерня).
    import numpy as np
    np.save("ndvi_raw.npy", ndvi_array)
    np.save("ndvi_mask.npy", mask_array)
    values = ndvi_array[mask_array > 0.5]
    print("\n   Распределение NDVI по пикселям (для настройки порогов):")
    edges = [-1, 0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60, 1.01]
    for lo, hi in zip(edges[:-1], edges[1:]):
        share = 100 * np.mean((values >= lo) & (values < hi))
        print(f"     [{lo:>5.2f} .. {hi:>4.2f})  {share:5.1f}%  " + "#" * int(share / 2))
    print(f"   Медиана {np.nanmedian(values):.3f}, p5 {np.nanpercentile(values, 5):.3f}, "
          f"p95 {np.nanpercentile(values, 95):.3f}")
    print("   Массивы сохранены: ndvi_raw.npy и ndvi_mask.npy — пришлите их вместе с ndvi_preview.png.")

    with open("ndvi_preview.png", "wb") as fh:
        fh.write(ndvi_raster_to_png(ndvi_array, mask_array))
    print("\n   Картинка слоя сохранена в ndvi_preview.png — откройте и проверьте, "
          "что контур похож на поле, а не на застройку.")

except Exception as e:
    import traceback
    print(f"\n   Растровый слой не получен: {type(e).__name__}: {e}")
    traceback.print_exc()
    print("   Временной ряд при этом работает — приложение запустится, "
          "но слой NDVI на карте будет недоступен.")
