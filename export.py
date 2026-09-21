"""
Формирование отчёта по полю в трёх форматах: CSV, GeoJSON и PDF.
"""

import csv
import io
import json
import os


def _register_cyrillic_font():
    """
    Регистрирует в reportlab шрифт с поддержкой кириллицы.

    Встроенные шрифты reportlab (Helvetica и прочие Base14) содержат только
    latin-1, поэтому без этой регистрации кириллица в PDF выводится чёрными
    прямоугольниками. DejaVu Sans поставляется внутри пакета matplotlib,
    который уже есть в зависимостях, — отдельная установка шрифтов не нужна.

    Возвращает: (обычный шрифт, полужирный шрифт) для c.setFont().
    """
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    if "DejaVuSans" in pdfmetrics.getRegisteredFontNames():
        return "DejaVuSans", "DejaVuSans-Bold"

    import matplotlib
    font_dir = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")

    pdfmetrics.registerFont(TTFont("DejaVuSans", os.path.join(font_dir, "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("DejaVuSans-Bold", os.path.join(font_dir, "DejaVuSans-Bold.ttf")))
    pdfmetrics.registerFont(TTFont("DejaVuSans-Oblique", os.path.join(font_dir, "DejaVuSans-Oblique.ttf")))

    return "DejaVuSans", "DejaVuSans-Bold"


def export_csv(enriched_timeseries: list[dict]) -> bytes:
    """CSV с временным рядом и отклонениями — удобно открыть в Excel."""
    buffer = io.StringIO()
    if not enriched_timeseries:
        return b""

    fieldnames = list(enriched_timeseries[0].keys())
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(enriched_timeseries)
    return buffer.getvalue().encode("utf-8")


def export_geojson(field_geojson: dict, summary: dict, field_name: str = "Поле №1",
                   zone_stats: dict | None = None, area_ha: float | None = None,
                   zones: list[dict] | None = None) -> bytes:
    """
    Контур поля со сводкой в properties. Открывается в QGIS и любой другой ГИС
    без конвертации; статистика зон попадает в те же properties, чтобы отчёт
    был самодостаточным.
    """
    feature = {
        "type": "Feature",
        "geometry": field_geojson,
        "properties": {
            "kind": "field",
            "field_name": field_name,
            "anomaly_share_pct": summary.get("anomaly_share_pct"),
            "anomaly_periods": summary.get("anomaly_periods"),
            "total_periods": summary.get("total_periods"),
            "latest_ndvi": summary.get("latest_point", {}).get("ndvi_mean")
            if summary.get("latest_point") else None,
            "latest_date": summary.get("latest_point", {}).get("date")
            if summary.get("latest_point") else None,
            "field_area_ha": area_ha,
        },
    }

    if zone_stats and "error" not in zone_stats:
        feature["properties"].update({
            "zone_problem_share_pct": zone_stats.get("problem_share_pct"),
            "zone_problem_area_ha": zone_stats.get("problem_area_ha"),
            "zone_mean_ndvi": zone_stats.get("field_mean_ndvi"),
            "zone_threshold_ndvi": zone_stats.get("threshold_ndvi"),
            "zone_coverage_pct": zone_stats.get("coverage_pct"),
        })
    features = [feature]
    for zone in zones or []:
        zone_feature = {**zone, "properties": {**zone["properties"], "field_name": field_name}}
        features.append(zone_feature)

    feature_collection = {"type": "FeatureCollection", "features": features}
    return json.dumps(feature_collection, ensure_ascii=False, indent=2).encode("utf-8")


def _render_chart_png(enriched_timeseries: list[dict], norm: float) -> bytes:
    """График NDVI/NDMI в PNG для встраивания в PDF. Бэкенд Agg работает
    без графического окна, что подходит для серверного рендеринга."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dates = [p["date"] for p in enriched_timeseries]
    ndvi = [p["ndvi_mean"] for p in enriched_timeseries]
    ndmi = [p.get("ndmi_mean") for p in enriched_timeseries]
    anomaly_x = [p["date"] for p in enriched_timeseries if p.get("is_anomaly")]
    anomaly_y = [p["ndvi_mean"] for p in enriched_timeseries if p.get("is_anomaly")]

    fig, ax = plt.subplots(figsize=(7, 3.3), dpi=150)
    ax.plot(dates, ndvi, marker="o", color="#2ca02c", label="NDVI", linewidth=1.5, markersize=3)
    ax.plot(dates, ndmi, marker="o", color="#1f77b4", label="NDMI", linewidth=1.5, markersize=3)
    has_history = any(p.get("baseline_kind") == "history" for p in enriched_timeseries)
    if has_history:
        base = [p.get("baseline_ndvi") for p in enriched_timeseries]
        std = [p.get("baseline_std") or 0 for p in enriched_timeseries]
        ax.fill_between(dates, [b - s for b, s in zip(base, std)],
                        [b + s for b, s in zip(base, std)],
                        color="gray", alpha=0.18, linewidth=0, label="Разброс прошлых сезонов")
        ax.plot(dates, base, color="gray", linestyle="--", linewidth=1,
                label="Норма по прошлым сезонам")
    else:
        ax.axhline(norm, color="gray", linestyle="--", linewidth=1, label="Норма сезона")
    if anomaly_x:
        ax.scatter(anomaly_x, anomaly_y, color="#d62728", marker="x", s=60, zorder=5, label="Аномалия")

    lowq = [p for p in enriched_timeseries if p.get("low_quality")]
    if lowq:
        ax.scatter([p["date"] for p in lowq], [p["ndvi_mean"] for p in lowq],
                   facecolors="none", edgecolors="#777777", marker="D", s=40, zorder=4,
                   label="Мало данных (облачность)")

    ax.set_ylabel("Значение индекса")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=3,
              fontsize=7, framealpha=0.9)
    ax.tick_params(axis="x", rotation=45, labelsize=6)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(alpha=0.3)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _wrap_to_width(canvas_obj, text: str, font: str, size: float, max_width: float) -> list[str]:
    """Переносит текст по словам так, чтобы каждая строка помещалась в max_width.
    Ширина считается по реальным глифам шрифта: кириллица шире латиницы,
    поэтому перенос по числу символов давал строки, вылезающие за поле."""
    lines, current = [], ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if canvas_obj.stringWidth(candidate, font, size) <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _build_recommendation(summary: dict) -> str:
    """Рекомендация агроному по последнему наблюдению: сигнал проверить
    участок на месте либо подтверждение, что ситуация в пределах нормы."""
    latest = summary.get("latest_point")
    if not latest:
        return "Недостаточно данных за период для рекомендации."
    if latest.get("is_anomaly"):
        return (f"Последнее наблюдение ({latest['date']}) показывает отклонение NDVI "
                f"на {abs(latest.get('deviation_pct', 0))}% ниже нормы сезона. "
                f"Рекомендуется выездная проверка участка агрономом в ближайшие дни "
                f"для уточнения причины (засуха, вредители, засорённость).")
    return (f"Последнее наблюдение ({latest['date']}) в пределах нормы сезона "
            f"(отклонение {latest.get('deviation_pct', 0)}%). Плановый мониторинг "
            f"можно продолжать в обычном режиме.")


def export_pdf_report(field_name: str, summary: dict, norm: float,
                      enriched_timeseries: list[dict],
                      zone_stats: dict | None = None,
                      area_ha: float | None = None,
                      zones: list[dict] | None = None) -> bytes:
    """PDF-отчёт по полю: сводка, график сезона, статистика проблемных зон,
    таблица аномальных периодов и рекомендация."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    font_regular, font_bold = _register_cyrillic_font()
    font_oblique = "DejaVuSans-Oblique"

    # ---------- Шапка ----------
    c.setFillColorRGB(0.16, 0.49, 0.20)  # тёмно-зелёный, агро-тематика
    c.rect(0, height - 1.6 * cm, width, 1.6 * cm, fill=1, stroke=0)
    c.setFillColorRGB(1, 1, 1)
    c.setFont(font_bold, 18)
    c.drawString(2 * cm, height - 1.15 * cm, "AgroBridge")

    y = height - 2.4 * cm
    c.setFillColorRGB(0, 0, 0)

    # ---------- Сводка ----------
    c.setFont(font_bold, 13)
    c.drawString(2 * cm, y, f"Поле: {field_name}")
    y -= 0.9 * cm

    c.setFont(font_regular, 10)
    summary_lines = []
    if area_ha:
        summary_lines.append(f"Площадь поля: {area_ha} га")
    summary_lines += [
        (f"Норма NDVI (по тем же декадам прошлых сезонов): {norm}"
         if summary.get("norm_mode") == "history"
         else f"Норма NDVI (среднее за текущий сезон, упрощённо): {norm}"),
        f"Всего периодов наблюдения: {summary.get('total_periods')}    "
        f"Периодов с аномалией: {summary.get('anomaly_periods')} "
        f"({summary.get('anomaly_share_pct')}%)",
    ]
    if summary.get("low_quality_periods"):
        summary_lines.append(
            f"Периодов с недостаточным покрытием снимком (облачность): "
            f"{summary['low_quality_periods']} — в расчёт аномалий не входят"
        )
    if zone_stats and "error" not in zone_stats:
        snapshot = f" (снимок {zone_stats['date']})" if zone_stats.get("date") else ""
        summary_lines.append(
            f"Проблемные зоны внутри поля{snapshot}: {zone_stats.get('problem_area_ha', '—')} га "
            f"({zone_stats.get('problem_share_pct')}% площади), "
            f"средний NDVI по полю {zone_stats.get('field_mean_ndvi')}"
        )
        summary_lines.append(
            f"Покрытие снимком: {zone_stats.get('coverage_pct')}% "
            f"(порог проблемной зоны: NDVI < {zone_stats.get('threshold_ndvi')})"
        )
    text_width = width - 4 * cm
    for line in summary_lines:
        for part in _wrap_to_width(c, line, font_regular, 10, text_width):
            c.drawString(2 * cm, y, part)
            y -= 0.5 * cm
        y -= 0.05 * cm

    y -= 0.3 * cm

    # ---------- График ----------
    try:
        chart_png = _render_chart_png(enriched_timeseries, norm)
        chart_reader = ImageReader(io.BytesIO(chart_png))
        chart_height = 7.2 * cm
        chart_width = width - 4 * cm
        c.drawImage(chart_reader, 2 * cm, y - chart_height, width=chart_width,
                    height=chart_height, preserveAspectRatio=True)
        y -= chart_height + 0.7 * cm
    except Exception:
        c.setFont(font_oblique, 9)
        c.drawString(2 * cm, y, "(график недоступен — не найден matplotlib)")
        y -= 0.8 * cm

    # ---------- Таблица аномалий ----------
    anomalies = [p for p in enriched_timeseries if p.get("is_anomaly")]

    c.setFont(font_bold, 12)
    c.drawString(2 * cm, y, "Периоды с отклонением ниже нормы")
    y -= 0.7 * cm

    if anomalies:
        col_x = [2 * cm, 5 * cm, 7.6 * cm, 10.2 * cm, 13 * cm]
        headers = ["Дата", "NDVI", "Норма", "NDMI", "Отклонение"]

        c.setFont(font_bold, 9)
        c.setFillColorRGB(0.16, 0.49, 0.20)
        c.rect(2 * cm, y - 0.15 * cm, width - 4 * cm, 0.6 * cm, fill=1, stroke=0)
        c.setFillColorRGB(1, 1, 1)
        for x, header in zip(col_x, headers):
            c.drawString(x, y, header)
        y -= 0.7 * cm

        c.setFont(font_regular, 9)
        c.setFillColorRGB(0, 0, 0)
        for i, point in enumerate(anomalies):
            if y < 2.5 * cm:
                c.showPage()
                y = height - 2 * cm
            if i % 2 == 0:
                c.setFillColorRGB(0.95, 0.95, 0.95)
                c.rect(2 * cm, y - 0.15 * cm, width - 4 * cm, 0.55 * cm, fill=1, stroke=0)
                c.setFillColorRGB(0, 0, 0)
            row = [point["date"], str(point["ndvi_mean"]), str(point.get("baseline_ndvi", "—")),
                   str(point.get("ndmi_mean", "—")), f"{point.get('deviation_pct')}%"]
            for x, value in zip(col_x, row):
                c.drawString(x, y, value)
            y -= 0.55 * cm
    else:
        c.setFont(font_oblique, 10)
        c.drawString(2 * cm, y, "Аномалий за период не выявлено.")
        y -= 0.7 * cm

    # ---------- Участки для обследования ----------
    if zones:
        y -= 0.4 * cm
        if y < 6 * cm:
            c.showPage()
            y = height - 2 * cm

        c.setFont(font_bold, 12)
        c.setFillColorRGB(0, 0, 0)
        c.drawString(2 * cm, y, "Участки внутри поля для обследования")
        y -= 0.7 * cm

        zcol_x = [2 * cm, 3.6 * cm, 6.6 * cm, 9.4 * cm, 12.4 * cm]
        zheaders = ["№", "Площадь, га", "NDVI зоны", "Широта", "Долгота"]
        c.setFont(font_bold, 9)
        c.setFillColorRGB(0.16, 0.49, 0.20)
        c.rect(2 * cm, y - 0.15 * cm, width - 4 * cm, 0.6 * cm, fill=1, stroke=0)
        c.setFillColorRGB(1, 1, 1)
        for x, header in zip(zcol_x, zheaders):
            c.drawString(x, y, header)
        y -= 0.7 * cm

        c.setFont(font_regular, 9)
        c.setFillColorRGB(0, 0, 0)
        for i, zone in enumerate(zones[:6]):
            if y < 2.5 * cm:
                c.showPage()
                c.setFont(font_regular, 9)
                y = height - 2 * cm
            zp = zone["properties"]
            if i % 2 == 0:
                c.setFillColorRGB(0.95, 0.95, 0.95)
                c.rect(2 * cm, y - 0.15 * cm, width - 4 * cm, 0.55 * cm, fill=1, stroke=0)
                c.setFillColorRGB(0, 0, 0)
            row = [str(zp["zone_id"]), str(zp.get("area_ha", "—")), str(zp["mean_ndvi"]),
                   str(zp["centroid_lat"]), str(zp["centroid_lon"])]
            for x, value in zip(zcol_x, row):
                c.drawString(x, y, value)
            y -= 0.55 * cm

    # ---------- Рекомендация ----------
    y -= 0.5 * cm
    if y < 3.5 * cm:
        c.showPage()
        y = height - 2 * cm

    c.setFont(font_bold, 12)
    c.drawString(2 * cm, y, "Рекомендация")
    y -= 0.7 * cm

    c.setFont(font_regular, 10)
    recommendation = _build_recommendation(summary)
    if zone_stats and "error" not in zone_stats and zone_stats.get("problem_share_pct", 0) >= 5:
        recommendation += (
            f" По снимку от {zone_stats.get('date', 'выбранной даты')} неоднородность внутри поля затрагивает "
            f"{zone_stats.get('problem_area_ha', '—')} га "
            f"({zone_stats.get('problem_share_pct')}% площади) — обследование стоит "
            f"начать именно с этих участков, они выделены на карте отдельным слоем."
        )
    for line in _wrap_to_width(c, recommendation, font_regular, 10, width - 4 * cm):
        if y < 2 * cm:
            c.showPage()
            c.setFont(font_regular, 10)
            y = height - 2 * cm
        c.drawString(2 * cm, y, line)
        y -= 0.5 * cm

    c.save()
    buffer.seek(0)
    return buffer.read()
