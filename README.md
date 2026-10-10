# AeroGraph — Planetary Climate Teleconnection Graph & Spectral Wavelets (2D)

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Leaflet](https://img.shields.io/badge/maps-Leaflet.js-199900.svg)](https://leafletjs.com/)
[![uPlot](https://img.shields.io/badge/charts-uPlot-blueviolet.svg)](https://github.com/leeoniya/uPlot)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Build: Passing](https://img.shields.io/badge/build-passing-brightgreen.svg)]()

**AeroGraph** — это высокопроизводительный исследовательский движок и веб-платформа для моделирования глобальной климатической системы Земли через призму **спектральной теории графов (Spectral Graph Theory)** и **спектральных графовых вейвлетов (SGWT)**.

Система обрабатывает синоптические поля геопотенциальной высоты изобарической поверхности 500 гПа ($Z_{500}$), выявляет макромасштабные атмосферные диполи (NAO, PNA, Сибирский антициклон), рассчитывает климатические индексы и строит авторегрессионные Lead-Lag прогнозы динамики тропосферы на горизонте 1–5 суток.

---

## 🌟 Ключевые возможности

- 🌐 **Интерактивная 2D-визуализация на Canvas & Leaflet**:
  - Высокопроизводительный рендеринг дуг телеконнекций квадратичными кривыми Безье с адаптивным зум-отсечением.
  - Дифференциация направления переноса: неоновый циан для опережающих связей ($\tau > 0$), янтарный для запаздывающих ($\tau < 0$), изумрудный для квазистационарных ($\tau = 0$).
  - Тепловая интерполяция волновых возмущений и спектральных вейвлетов.

- 🔬 **Спектральные графовые вейвлеты (SGWT)**:
  - Вычисление непрерывных вейвлет-атомов на нерегулярных геодезических сетках с помощью быстрого рекуррентного разложения Чебышева 30-го порядка (алгоритм Кленшоу) без прямого обращения матриц.
  - Анализ пространственной диссипации синоптической энергии на масштабах $s \in [2.0, 10.0]$.

- ⚡ **Векторизованный телеконнекционный движок**:
  - Кросс-корреляционный анализ временных рядов с лагами $\tau \in [-7, +7]$ суток.
  - Фильтрация локального шума: пространственная отсечка $d_{\min} \ge 1\,000\text{ км}$ (расчет по ортодромии гаверсинусов) и статистический порог $|r| \ge 0.65$ с поправкой Кенуя/Дэвиса на эффективные степени свободы $N_{\mathrm{eff}}$.

- 📈 **Реальное время: Планетарные индексы и Lead-Lag прогноз**:
  - Расчёт индекса Североатлантического колебания (**NAO**) через нормализованную разность аномалий Азорского максимума и Исландской депрессии.
  - Расчёт индекса Арктического колебания (**AO**) через скалярную проекцию поля $Z_{500}$ на ведущую нетривиальную гармонику Лапласиана (собственный вектор $v_1$).
  - 5-дневная авторегрессионная модель траектории узла сетки с расчетом доверительных интервалов $\pm 1\sigma$ и синоптическим вердиктом.

- 📊 **Автоматизированный аналитический экспорт (ZIP-дамп)**:
  - Генерация 7 научных графиков высокого разрешения (300 DPI, тёмная академическая палитра).
  - Сводные отчеты: `REPORT.md` и автономный интерактивный дашборд `REPORT.html` с поддержкой экспорта в PDF через системную печать.
  - Табличные наборы данных: `nodes_summary.csv`, `significant_edges.csv`, `climate_indices.json`, `nodes_predictability.csv`.

---

## 🏗️ Архитектура системы

```
                 [ Open-Meteo ERA5 / NOAA GFS Reanalysis ]
                                     │
                                     ▼
                        [ scripts/fetch_data.py ]
                                     │
                              data/cache/grid.npz (T=60, 13x72)
                                     │
               ┌─────────────────────┴─────────────────────┐
               ▼                                           ▼
   [ engine/teleconnection.py ]                  [ engine/forecast.py ]
   • Lagged Cross-Correlation (tau: -7..+7)      • Macro Indices: NAO & AO
   • Great-Circle Distance Filter (d >= 1000km)  • Lead-Lag 5-day Autoregression
   • Effective DoF (Quenouille adjustment)       • R² Predictability Skill
               │                                           │
               ▼                                           │
      data/teleconnections.db (SQLite)                     │
   (graph_nodes & teleconnection_edges)                    │
               │                                           │
               ▼                                           │
       [ engine/spectral.py ]                              │
   • Normalized Graph Laplacian L                          │
   • Sparse Eigensolver (Fiedler value λ₂)                 │
   • Chebyshev SGWT Wavelet Polynomials                    │
               │                                           │
               └─────────────────────┬─────────────────────┘
                                     ▼
                            [ api/main.py ]
                           (FastAPI REST API)
                                     │
           ┌─────────────────────────┴─────────────────────────┐
           ▼                                                   ▼
  [ engine/export.py ]                              [ client/ (Vite/Leaflet/uPlot) ]
  • 7 Charts (300 DPI)                              • 2D Canvas Arcs Overlay
  • REPORT.md & REPORT.html                         • Dynamic SGWT Heatmap Layer
  • ZIP Export Stream                               • Forecast Drawer & Trajectory SVG
```

---

## 📐 Математический аппарат

### 1. Нормированный симметричный Лапласиан графа
Для графа климатической сети с взвешенной матрицей смежности $A$ ($A_{ij} = |r_{ij}|$ при $d_{ij} \ge 1000\text{ км}$) и диагональной матрицей степеней узлов $D_{ii} = \sum_j A_{ij}$:
$$\mathcal{L} = I - D^{-1/2} A D^{-1/2}$$
Спектр $\mathcal{L}$: $0 = \lambda_1 \le \lambda_2 \le \dots \le \lambda_N \le 2$.  
$\lambda_2$ (**число Фидлера**) характеризует алгебраическую связность тропосферного волновода, а вектор Фидлера $v_2$ осуществляет оптимальную спектральную бипартицию атмосферы.

### 2. Спектральные графовые вейвлеты (SGWT)
Вейвлет-оператор в вершине $i$ на масштабе $s$:
$$\psi_{s, i} = g(s\mathcal{L}) \delta_i \approx \sum_{m=0}^{M} c_m(s) T_m(\tilde{\mathcal{L}}) \delta_i$$
где $T_m(x)$ — ортогональные многочлены Чебышева 1-го рода, удовлетворяющие трехчленной рекуррентной формуле:
$$T_0(x) = 1, \quad T_1(x) = x, \quad T_{m}(x) = 2x T_{m-1}(x) - T_{m-2}(x)$$
а $\tilde{\mathcal{L}} = \frac{2}{\lambda_{\max}} \mathcal{L} - I$ масштабирует спектр в канонический отрезок $[-1, 1]$.

### 3. Эффективные степени свободы (Поправка Кенуя/Дэвиса)
Для компенсации сильной автокорреляции временных рядов давления $Z_{500}$ статистическая значимость $p$-значений рассчитывается на основе скорректированного числа степеней свободы:
$$N_{\mathrm{eff}} = \frac{N}{1 + 2 \sum_{k=1}^{K} \rho_x(k) \rho_y(k)}$$

---

## 🚀 Быстрый старт

### 1. Клонирование репозитория
```bash
git clone https://github.com/sbuw/aerograph.git
cd aerograph
```

### 2. Настройка бэкенда (Python 3.11+)
```bash
# Создание и активация виртуального окружения
python -m venv .venv
source .venv/bin/activate  # На Linux/macOS
# .venv\Scripts\activate   # На Windows

# Установка зависимостей
pip install -r requirements.txt
```

### 3. Инициализация данных и расчет математического графа
```bash
# 1. Загрузка синоптических полей Z500 (Open-Meteo ERA5 / Reanalysis)
python scripts/fetch_data.py

# 2. Расчет телеконнекций, фильтрация по ортодромии и сохранение в SQLite
python scripts/build_graph.py

# 3. Спектральная декомпозиция Лапласиана и пре-кэширование вейвлетов SGWT
python scripts/compute_spectral.py

# (Опционально) Комплексная разовая инициализация базы данных и спектра:
python scripts/init_db.py
```

### 4. Запуск сервера FastAPI
```bash
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```
API документация Swagger UI доступна по адресу: `http://127.0.0.1:8000/docs`.

### 5. Запуск веб-клиента (Node.js 18+)
```bash
cd client
npm install
npm run dev
```
Откройте браузер по адресу: `http://localhost:5173`.

### 6. Проверка эндпоинтов и верификация API (CLI)
```bash
python scripts/test_api.py
```

---

## 📡 REST API Reference

| Метод | Эндпоинт | Описание |
|:---|:---|:---|
| `GET` | `/api/health` | Проверка доступности базы SQLite, предзагруженного Лапласиана и кэшей в RAM. |
| `GET` | `/api/nodes` | Список всех 936 узлов геодезической сетки $5^\circ \times 5^\circ$ с координатами и степенями. |
| `GET` | `/api/edges` | Активные ребра графа телеконнекций с фильтрацией по порогу `min_r` (по умолчанию 0.65). |
| `GET` | `/api/node/{id}/teleconnections` | Все инцидентные связи конкретного выбранного узла с распределением лагов. |
| `GET` | `/api/spectrum` | Спектр собственных значений Лапласиана $\lambda_k$, число Фидлера и спектральный радиус. |
| `GET` | `/api/wavelet` | Динамический расчет вейвлет-отклика Чебышева для узла `node_id` на масштабе `scale`. |
| `GET` | `/api/climate-indices` | Текущие макроиндексы NAO и AO, фазовые знаки, отклонения в $\sigma$ и синоптическое описание. |
| `GET` | `/api/node/{id}/forecast` | 5-дневный Lead-Lag авторегрессионный прогноз изобарической высоты с полосой $\pm 1\sigma$. |
| `GET` | `/api/export/report` | Формирование и потоковая отдача полного аналитического ZIP-архива с отчетами и графиками. |

---

## 📑 Генерация аналитического отчета через CLI

Вы можете сгенерировать полный исследовательский пакет без запуска веб-интерфейса:

```bash
python scripts/export_report.py
```

Сформированный архив сохраняется в директорию `exports/climate_report_<YYYY-MM-DD>.zip` и содержит:
- `REPORT.html` — автономный адаптивный темный дашборд (CSS Glassmorphism, Print-ready для сохранения в PDF).
- `REPORT.md` — исчерпывающее аналитическое описание с формулами и выводами.
- `charts/` — 7 графиков высокого разрешения (300 DPI):
  1. `01_teleconnection_map.png` — глобальная карта ведущих тропосферных волноводов.
  2. `02_laplacian_spectrum.png` — спектр Лапласиана с отметкой алгебраической связности $\lambda_2$.
  3. `03_distance_vs_correlation.png` — диаграмма ортодромического расстояния против $|r|$ с квадратичным трендом.
  4. `04_lag_distribution.png` — гистограмма распределения временных лагов (опережение/синхронность/запаздывание).
  5. `05_wavelet_energy_decay.png` — профиль затухания пространственной энергии вейвлетов Чебышева.
  6. `06_climate_indices_phases.png` — диаграмма текущих фаз климатических индексов NAO и AO.
  7. `07_forecast_lead_time_skill.png` — кривая точности $R^2$ авторегрессионной модели на горизонте 1–5 суток.
- `data/` — табличные выгрузки:
  - `nodes_summary.csv`
  - `significant_edges.csv`
  - `summary_metrics.json`
  - `climate_indices.json`
  - `nodes_predictability.csv`

---

## 📜 Лицензия

Проект распространяется под открытой лицензией [MIT](LICENSE).
