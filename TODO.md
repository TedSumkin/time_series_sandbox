# TODO — Pet Sandbox: Task × Model (4 недели)

Цель: собрать песочницу, где в 1 команду запускается эксперимент **данные → baseline(ы) → модель → метрики/графики → артефакты**, с “предохранителями” от тихих ошибок и с репродьюсибилити.
При этом задачи должны быть произвольными, типа:

- forecasting
- classification
- reconstruction
- anomaly detection
- changepoint detection
Definition of Done (общий):

- Один CLI/entrypoint запускает train+eval и сохраняет артефакты (конфиг, метрики, графики, чекпоинт).
- Есть 4 baseline’а “под рукой”.
- Есть 10 предохранителей (sanity checks + pytest), которые падают быстро и понятно.
- Добавление новой модели или новой задачи требует минимальных правок (по контракту интерфейсов).

---

## Недели 1–2 — Каркас + 1 задача (Forecasting) + первые предохранители

### День 2: Скелет репо и “контракт интерфейсов”

- [x] Создать репозиторий, структуру папок:
  - `sandbox/`
    - `tasks/`
    - `models/`
    - `baselines/`
    - `runner/`
    - `data/`
    - `metrics/`
    - `viz/`
    - `utils/`
  - `tests/`
  - `configs/`
  - `README.md`
- [x] Зафиксировать “контракт интерфейсов” (в README или `sandbox/interfaces.md`):
  - Task:
    - `build_dataloaders(cfg) -> (train, val, test)`
    - `loss_fn(outputs, batch) -> loss`
    - `metrics(pred, batch) -> dict[str, float]`
    - `visualize(pred, batch, outdir) -> None`
    - `sanity_checks(cfg) -> list[CheckResult]`
  - Model:
    - `forward(batch) -> outputs`
    - `predict(batch) -> pred`
    - `configure_optimizers(cfg) -> (optimizer, scheduler|None)`
  - Runner:
    - `train(task, model, dataloaders, cfg) -> artifacts`
    - `eval(task, model, dataloader, cfg) -> metrics + optional preds`
    - `test(task, model, dataloader, cfg) -> metrics + optional preds`

- [ ] Принять формат batch (минимум для forecasting):
  - `batch = { "x": (B,T,F), "y": (B,H,F_out), "meta": {...} }` (или эквивалент)

### День 4: Runner v0 (train/eval, логи, чекпоинты, артефакты)

- [ ] Реализовать `runner/runner.py`:
  - [ ] train loop + val loop
  - [ ] early stopping (простая версия)
  - [ ] сохранение best checkpoint
  - [ ] сохранение `metrics.json`/`metrics.csv`
  - [ ] сохранение `config.yaml` + `seed`
- [ ] Добавить базовое логирование (stdout + файл `run.log`)
- [ ] Добавить “sanity run” режим: `--sanity_steps N` (например 100)

### День 6: Task v0 — Forecasting на простом датасете

- [ ] Выбрать стартовый датасет:
  - вариант 1: синтетика (быстро, без внешних зависимостей)
  - вариант 2: публичный маленький (ETT/Traffic/Electricity) — только если уже удобно тянуть
- [ ] Реализовать `tasks/forecasting.py`:
  - [ ] генерация окон (context length `T`, horizon `H`)
  - [ ] dataloaders (train/val/test)
  - [ ] метрики: MSE/MAE + horizon-wise MSE (по шагам горизонта)
  - [ ] визуализация: `pred_vs_true.png`, `horizon_errors.png`

### День 8: 2 baseline’а (контрольные точки)

- [ ] Baseline #1: Persistence/Naive (последнее значение / last-window)
- [ ] Baseline #2: Ridge regression по лагам (sklearn) или простой linear head в torch
- [ ] Убедиться, что baseline’ы используют тот же Task/Runner контракт (через адаптер)

### День 10: Предохранители v1 (минимум 5) + pytest каркас

- [ ] Сделать механизм checks: `utils/checks.py` + формат результата (ok/fail + message)
- [ ] Реализовать 5 sanity checks для forecasting:
  1. [ ] `finite_check`: нет NaN/Inf в x/y, loss, grads
  2. [ ] `shape_check`: формы `pred` и `y` согласованы
  3. [ ] `split_check`: нет пересечения индексов train/val/test (если индексы доступны)
  4. [ ] `overfit_one_batch_check`: модель за N шагов заметно снижает loss на одном batch
  5. [ ] `off_by_one_check` на синтетике (или через контролируемый сдвиг)
- [ ] Добавить `tests/test_sanity.py`:
  - [ ] тесты на падение при намеренной поломке (1–2 негативных теста)

### День 12–14: Полировка Week 2 (первый “готовый станок”)

- [ ] CLI/entrypoint:
  - `python run.py task=forecast model=mlp baseline=persistence`
- [ ] Документировать “Как добавить модель за 10 минут”
- [ ] Зафиксировать “Definition of Done Week 2”:
  - [ ] 1 задача (forecasting) + 2 baseline + 1 простая нейромодель (MLP) + 5 checks
  - [ ] артефакты сохраняются стабильно
  - [ ] sanity-run проходит за несколько минут

---

## Недели 3–4 — Расширение: 4 baseline’а + 10 предохранителей + 2-я задача

### День 16: Ещё 2 baseline’а

- [ ] Baseline #3: EWMA / Moving Average (как фильтр или как прогноз по экспон. сглаживанию)
- [ ] Baseline #4: Small CNN/TCN-lite (быстрый нейро-бейзлайн)
- [ ] Сравнение baseline’ов в единой таблице результатов

### День 18: Предохранители v2 (довести до 10)

Добавить ещё 5:
  6. [ ] `leakage_check`: нет будущих точек во входе относительно таргета (по time index / метаданным)
  7. [ ] `determinism_check`: 2 запуска sanity-run с одним seed дают близкие результаты (допуск)
  8. [ ] `checkpoint_roundtrip_check`: save/load не меняет pred на одном batch
  9. [ ] `baseline_sanity_check`: baseline не даёт “подозрительно идеальные” метрики (MSE≈0) без причины
  10. [ ] `speed_check`: sanity-run (например 100 шагов) не превышает порог времени (на твоей машине)

### День 20: Оформить checks как pytest + удобные сообщения

- [ ] Перевести checks в `tests/`:
  - [ ] позитивные тесты (всё проходит)
  - [ ] негативные тесты (намеренно ломаем и ожидаем fail)
- [ ] Сделать сообщения checks максимально диагностическими (“что именно не так”)

### День 22: Вторая задача (минимальная) — на выбор

Выбрать 1:

- вариант A: `ClassificationTask` (окна → класс)
- вариант B: `AnomalyTask` (point-wise или window-wise)
- вариант C: `RULTask` (регрессия по окнам)

TODO:

- [ ] Реализовать `tasks/<chosen>.py` по тому же контракту:
  - [ ] dataloaders
  - [ ] метрики (F1/AUROC или PR-AUC; для RUL MAE/RMSE)
  - [ ] 1 визуализация (confusion matrix / anomaly score plot / pred vs true)

### День 24: Универсализация формата batch (без боли)

- [ ] Определить минимально-универсальный формат `batch`:
  - `x`, `y`, `mask` (опционально), `time_idx` (опционально), `id` (опционально)
- [ ] Обновить Runner, чтобы он не зависел от конкретной задачи (только от контракта)
- [ ] Обновить baseline-адаптеры, чтобы работали с любой задачей (где применимо)

### День 26: Репродьюсибилити и “таблица результатов”

- [ ] Автоматический отчёт после запуска:
  - [ ] `results.csv` (run_id, task, model, baseline, seed, метрики)
  - [ ] папка `runs/<run_id>/` с артефактами
- [ ] Сохранение “environment snapshot”:
  - [ ] `pip freeze`/`conda env export` (по желанию)
  - [ ] `git hash` + dirty flag

### День 28: README “как продукт” + финальная проверка

- [ ] README:
  - [ ] Quickstart (1 команда)
  - [ ] Как добавить `Task`
  - [ ] Как добавить `Model`
  - [ ] Список baseline’ов
  - [ ] Список предохранителей
- [ ] Финальный критерий “готово Week 4”:
  - [ ] 2 задачи (forecasting + выбранная)
  - [ ] 4 baseline’а
  - [ ] 10 предохранителей (pytest + runtime)
  - [ ] стабильные артефакты и таблица результатов

---

## Бэклог (если останутся силы, НЕ обязательно)

- [ ] Hydra/omegaconf конфиги
- [ ] Optuna runner (30 трейлов) с сохранением лучших конфигов
- [ ] Простая web-страница/markdown-репорт с картинками из последнего run
- [ ] Dockerfile / Makefile

---

## Каноническая команда запуска (целевое UX)

- `python run.py task=forecast model=mlp data=synthetic horizon=24 context=96 seed=42`
- `python run.py task=forecast baseline=persistence`
- `pytest -q` (предохранители)

---

- [ ] сделать нормальные dummy-configs

