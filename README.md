# Kazakh Model Watch

Небольшой benchmark AI-моделей на казахском языке с вручную проверенными заданиями и сохранёнными ответами. Приоритет MVP — минимальный работающий pipeline и реальные измерения.

**Статус:** Dataset Loader, Scoring Engine, Model Adapter Foundation и OpenRouter Integration. Есть OpenRouter adapter и CLI для одного smoke-запроса с явным `--live`; 2026-10-15 выполнен один подтверждённый вызов `qwen/qwen3.5-9b`, получен ответ `5`. Все пять заданий остаются черновиками. Evaluation Runner, полный benchmark и GitHub Actions ещё предстоит реализовать; результатов полного сравнения пока нет.

Основная техническая спецификация — [docs/MVP_SPEC.md](docs/MVP_SPEC.md). Реализация и критерии готовности определяются этим документом.

## Приватность и публикация

[GitHub-репозиторий](https://github.com/gulbaur/Kazakh-Model-Watch) должен оставаться **PRIVATE** до завершения MVP и release review. Публичный релиз — отдельный шаг после review и решения владельца; успешный запуск benchmark сам по себе не разрешает менять visibility или публиковать артефакты.

До будущего публичного релиза необходимо:

- Выполнить acceptance criteria из спецификации: вручную проверенный датасет, реальные API-ответы минимум двух моделей, метрики, сохранённые артефакты и работающий CI.
- Проверить содержимое файлов и историю Git на API-ключи, `.env`, конфиденциальные сведения и данные работодателя. При обнаружении ключа отозвать его; одного удаления файла недостаточно.
- Проверить происхождение и права использования каждого задания, корректность scoring и воспроизводимость отчёта.
- Выбрать проверенный завершённый запуск, проверить все его артефакты перед включением в Git и описать ограничения малого benchmark.
- Зафиксировать итог release review и разрешение владельца на публичную публикацию. До этого visibility остаётся PRIVATE.

Эти условия применяются также к README, результатам, GitHub Releases и другим публичным материалам проекта.

## План MVP

- 24 задания: по 8 на `numbers`, `dates`, `morphology`.
- Минимум две различные реальные текстовые модели через официальные API, один общий prompt и профиль генерации.
- Accuracy, строгий exact match, latency и coverage; стоимость только при достоверных данных, иначе `null`/`N/A`.
- Ответы и метрики в JSON/CSV, Markdown leaderboard и пересчёт отчёта без сети.
- Один последовательный процесс и небольшие функции/модули для данных, API-вызовов, scoring и экспорта. Без сервера, базы данных, framework плагинов и дополнительных уровней абстракции.
- `pytest` и GitHub Actions без реальных API-вызовов на push/PR. Реальное сравнение запускается отдельно вручную.

## Текущая структура

```text
kazakh-model-watch/
├── config/
│   └── openrouter_model.json # публичный снимок возможностей выбранной модели
├── data/
│   ├── benchmark.json       # пустая исходная заготовка; не используется loader
│   └── benchmark.jsonl      # пять черновых заданий, review.status=draft
├── docs/
│   └── MVP_SPEC.md          # основная техническая спецификация
├── src/
│   ├── .gitkeep
│   └── kazakh_model_watch/
│       ├── __init__.py
│       ├── dataset.py       # загрузка и валидация JSONL
│       ├── model_adapter.py # ModelRequest, ModelResponse, интерфейс и offline mock
│       ├── normalization.py # общая нормализация kk-short-v1
│       ├── openrouter_adapter.py
│       ├── openrouter_smoke.py
│       └── scoring.py       # exact match и normalized correctness
├── tests/
│   ├── .gitkeep
│   ├── test_dataset.py
│   ├── test_model_adapter.py
│   ├── test_openrouter_adapter.py
│   └── test_scoring.py
├── .gitignore
├── LICENSE
├── README.md
└── pyproject.toml
```

`data/benchmark.jsonl` содержит черновики для разработки: два задания numbers, два dates и одно morphology. Эталоны — кандидаты для будущей ручной проверки. API-конфигурация, `results/<run_id>/` и `LEADERBOARD.md` появятся на соответствующих этапах.

## Локальное окружение

Python **3.11+**. Обычный `venv` и `pip` достаточны; отдельный менеджер окружения не требуется. `pyproject.toml` фиксирует прямые зависимости: setuptools, pytest, HTTPX для HTTP и python-dotenv для явной локальной загрузки ключа. SDK провайдера не используется.

Из корня репозитория, PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Linux/macOS:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
```

Активация окружения не обязательна: команды используют его интерпретатор напрямую. Установка добавляет пакет и зависимости; полный benchmark ещё не реализован. Конфигурация setuptools использует существующий `src/`, тесты находятся в `tests/`.

Запуск тестов из корня:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

На Linux/macOS использовать `.venv/bin/python -m pytest`. Тесты проверяют контракт датасета, включая некорректные данные и финальный режим; они не доказывают языковую правильность черновиков. Метаданные `verified` в тестовых fixtures синтетические и не являются отметками реальной проверки. Команды `validate`, `run` и `report` будут документированы вместе с их реализацией.

## Dataset Loader

```python
from kazakh_model_watch.dataset import load_dataset

# Разработка: любой непустой набор, черновики разрешены.
drafts = load_dataset("data/benchmark.jsonl")

# Финальная проверка: 24 verified-задания, по 8 на категорию.
# Для текущих пяти черновиков этот вызов ожидаемо завершится ошибкой.
final_dataset = load_dataset("data/benchmark.jsonl", final=True)
```

Loader возвращает список JSON-объектов, сохраняя порядок и исходные строки. Он проверяет обязательные поля и типы, уникальность ID, категории и типы ответов, канонические целые, реальные календарные даты, одно слово в казахской кириллице и метаданные источника/review. Для допустимых вариантов проверяются включение эталона и отсутствие дубликатов после NFC, нормализации пробелов и casefold. Scoring и семантическая оценка здесь не выполняются.

По условиям Dataset Foundation в режиме разработки допускается `review.status="draft"` с `reviewer_id=null`, `reviewed_at=null` и непустым `note`. Остальные обязательные поля сохраняются. `verified` в любом режиме требует непустой reviewer ID и корректной даты; ставить эту отметку можно только после реальной проверки человеком. `final=True` применяет полные требования размера, баланса и review из [MVP_SPEC.md](docs/MVP_SPEC.md). Спецификация на этом этапе не изменяется.

Ошибки данных — `DatasetValidationError` (подкласс `ValueError`), с файлом, физическим номером строки и полем для ошибок записи. Ошибки доступа к файлу остаются `OSError`. Читается UTF-8 без BOM, пустые строки пропускаются, JSON-массив вместо JSONL и повторяющиеся ключи JSON отклоняются. Сетевые запросы и дополнительные runtime-зависимости отсутствуют.

## Scoring Engine

```python
from kazakh_model_watch.scoring import score_response

scores = score_response(
    raw_response=" 5\n",
    reference_answer="5",
    accepted_answers=["5"],
    status="ok",
)
# {"exact_match": 0, "correct": 1}
```

Эталоны передаются из проверенного loader датасета. `exact_match` сравнивает полный ответ с `reference_answer` через равенство Python Unicode-строк, без преобразований. `correct` сравнивает полный ответ со всеми `accepted_answers` после общей нормализации `kk-short-v1`: NFC → удаление пробелов по краям и свёртка внутренних Unicode whitespace → casefold. Эта же функция используется loader для проверки дубликатов вариантов.

Любой `status != "ok"` даёт нули по обеим оценкам, даже при совпадающем тексте. `None`, пустой и пробельный текст также дают нули. Для `status="ok"` текст другого типа вызывает `TypeError`; автоматического преобразования в строку нет. Числа из объяснений не извлекаются, Markdown и пунктуация не удаляются, похожие буквы и zero-width символы сохраняются. Исходные ответы и эталоны не изменяются. Scoring работает локально, без API; агрегация, runner и экспорт на этом этапе не реализованы.

## Model Adapter Foundation

`ModelRequest` и `ModelResponse` — dataclass-структуры с проверкой типов и значений при создании. Их поля сохраняют исходный prompt и видимый ответ; атрибуты нельзя переназначать. В запросе — provider, model ID, полный prompt и параметры; defaults из спецификации: temperature `0`, max output tokens `128`, timeout `60` секунд. Prompt собирается вызывающим кодом; адаптер не добавляет инструкции и не получает эталоны.

Ответ содержит восемь полей: `status`, `raw_response`, `requested_model_id`, `returned_model_id`, `request_id`, `finish_reason`, `usage`, `latency_ms`. Неизвестные metadata/usage/latency остаются `None`; нулевые счётчики и latency не заменяются на `None`. У `ok` должен быть непробельный текст, а ошибки и обрывы имеют `latency_ms=None`, поскольку успешная попытка отсутствует. `usage` содержит только неотрицательные целочисленные счётчики токенов; их значения не вычисляются из длины текста.

`ModelAdapter` задаёт единственный синхронный метод `generate(request) -> response` и сам использует только стандартную библиотеку. Плагинов и registry нет; конкретная интеграция OpenRouter описана ниже. Таймер и повторы полного benchmark, `attempts`, `error` и `total_elapsed_ms` относятся к будущему runner; на этом этапе он не реализуется.

Offline-пример с явно синтетическим ответом:

```python
from kazakh_model_watch.model_adapter import MockAdapter, ModelRequest, ModelResponse

request = ModelRequest(provider="mock", model_id="mock-model", prompt="Тек жауапты жазыңыз.")
preset = ModelResponse(status="ok", raw_response="5", requested_model_id="mock-model")
response = MockAdapter(preset).generate(request)
```

Mock возвращает копию заранее заданного `ModelResponse`, без сети, задержек и измерения времени. Для сценариев timeout/API error задаются соответствующий статус и `raw_response=None`; для пустого ответа — `empty_response`. Тип и статус ошибочного ответа не меняются автоматически. Mock принимает только `provider="mock"` и model ID своего preset; шаблон и результаты отдельных вызовов не разделяют изменяемый словарь usage.

Ответы, ID, usage и latency mock являются тестовыми fixtures, а не измерениями реальной модели. Провайдер `mock` и его результаты должны исключаться из любого реального сравнения и leaderboard. API-ключи не нужны и не входят в структуры запроса/ответа.

## OpenRouter: один smoke test

`OpenRouterAdapter` использует [Chat Completions API](https://openrouter.ai/docs/api/api-reference/chat/create-a-chat-completion) и существующий `ModelAdapter`. Передаётся полный prompt из `ModelRequest`, один user message, temperature `0`, max tokens `128`, timeout `60`, stream `false`. Настройки не меняются автоматически; профиль или модель, не соответствующие metadata, дают безопасный код ошибки до платного вызова.

Поддержка temperature/max_tokens для `qwen/qwen3.5-9b` проверена через публичный [Models API](https://openrouter.ai/api/v1/models); выбранные поля сохранены в [config/openrouter_model.json](config/openrouter_model.json) с датой проверки. Снимок загружается локально, поэтому во время генерации дополнительных GET нет. `provider.require_parameters=true` требует поддержки параметров конкретным backend; `allow_fallbacks=false` отключает fallback провайдера согласно [правилам маршрутизации](https://openrouter.ai/docs/guides/routing/provider-selection). Список запасных моделей не передаётся.

Offline-проверка команды из корня проекта — без сети, ключа и чтения `.env`:

```powershell
.\.venv\Scripts\python.exe -m kazakh_model_watch.openrouter_smoke
```

После отдельного подтверждения одного платного запроса:

```powershell
.\.venv\Scripts\python.exe -m kazakh_model_watch.openrouter_smoke --live --dotenv
```

`--dotenv` читает только локальный `.env` при наличии `--live` и не перезаписывает уже заданные переменные окружения. Адаптер берёт ключ только из `OPENROUTER_API_KEY`. Без `--dotenv` ключ необходимо заранее задать в окружении; команда и её вывод не содержат значения ключа. Из Linux/macOS использовать `.venv/bin/python` вместо Windows-пути.

Smoke выполняет ровно один POST для фиксированной модели и задания: «Екіге үшті қосқанда қанша болады? Жауапты тек бүтін санмен жазыңыз.» HTTP redirects и retries отключены. Команда не загружает весь датасет, не пишет leaderboard и не запускается автоматически в pytest/CI. Unit-тесты, включая проверку ветки `--live`, используют исключительно `httpx.MockTransport`, синтетический ключ и подменённую загрузку dotenv.

Вывод содержит поля `ModelResponse`, измеренный `elapsed_ms`, диагностический `http_status` и статический `error_code`. `elapsed_ms` измеряет весь вызов generate, включая HTTP и преобразование ответа; до HTTP-попытки остаётся `None`. `ModelResponse.latency_ms` остаётся `None`: измерение latency для benchmark относится к будущему runner. Код выхода `0` означает ok, `1` — ошибку, пустой или обрезанный ответ. Raw text не нормализуется и не оценивается этой командой.

HTTP ошибки, timeout, пустой/обрезанный ответ и некорректная схема обрабатываются явно. Текст исключений, тела ошибок и Authorization header не выводятся. Если сервер отражает ключ в JSON, ответ отклоняется целиком с `sensitive_response`. `usage` сохраняет только сообщённые API целочисленные счётчики токенов; вложенные token details представлены через dotted keys. Пропущенные счётчики не вычисляются, cost-поля не превращаются в стоимость или токены.

Реальная проверка 2026-10-15: ровно один подтверждённый POST, HTTP `200`, status `ok`, requested/returned model ID `qwen/qwen3.5-9b`, raw response `5`, finish reason `stop`, измеренный `elapsed_ms=7116.382699459791`. API сообщил prompt/completion/total tokens `83/293/376`, включая `291` reasoning token. Вывод сохранён локально в игнорируемом `logs/`; это проверка интеграции на одном задании, без результатов benchmark и leaderboard. Стоимость не рассчитывалась.

При переданном `max_tokens=128` API сообщил `completion_tokens=293`. Счётчики сохранены как получены; причина расхождения для выбранного backend не установлена. По [документации OpenRouter](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens#reasoning-tokens-and-max_tokens), у большинства провайдеров reasoning и видимый ответ делят этот лимит. Перед полноценным сравнением нужно проверить фактическую семантику лимита для этого endpoint. Параметры запроса и спецификация по результату smoke не изменялись, дополнительных вызовов не было.

Ограничения: снимок Models API может устареть; require_parameters проверяется при маршрутизации, но доступность и quota подтверждаются только реальным вызовом. Alias модели может изменяться, возвращённый ID сохраняется. Поддерживается один текстовый completion; tool calls, массивы content и неожиданные модели отклоняются. HTTPX timeout задаётся для сетевых фаз, отдельного общего deadline нет. Provider reasoning defaults не меняются; finish_reason=length всегда даёт truncated. CLI рассчитан на локальную editable-установку и присутствие config-файла; это не Evaluation Runner.

## Ключи и локальные данные

Ключи читаются из переменных окружения; в будущем GitHub Actions будет использовать GitHub Secrets. Для offline-тестов ключи не нужны. `.env` и его варианты, файлы ключей, `secrets/`, локальные данные и `results/` игнорируются Git. `.env.example` должен содержать только пустые значения или явные placeholders; настоящий ключ в шаблон не добавляется.

Запрещено добавлять API-ключи, `.env`, конфиденциальные данные или материалы работодателя даже в PRIVATE-репозиторий. Используются только оригинальные задания либо разрешённые публичные источники. Проверенный датасет и документация остаются доступными для включения в Git; общие шаблоны `*.json` и `*.csv` не игнорируются.

`results/` исключён по умолчанию, чтобы ответы и служебные сведения не попадали в Git случайно. Перед адресным включением конкретного запуска проверить prompts, responses, errors, config и manifest; запрет на секреты продолжает действовать. Финальные проверенные артефакты сохраняются в репозитории согласно спецификации.

Перед коммитом проверять состав и содержимое изменений:

```bash
git status --short
git diff
git diff --cached
```

Новые файлы нужно просмотреть отдельно: `git diff` не показывает untracked-файлы. `.gitignore` предотвращает случайное добавление известных локальных файлов, но не проверяет содержимое и не удаляет уже отслеживаемые секреты; см. [документацию Git](https://git-scm.com/docs/gitignore).

## Лицензия

[MIT](LICENSE) выбрана как простая стандартная лицензия для кода и документации проекта; текст соответствует [MIT License](https://opensource.org/license/mit). Для заимствованных заданий отдельно сохраняются источник и условия использования. Наличие LICENSE не меняет PRIVATE-стратегию и условия release review.

В `pyproject.toml` добавлен классификатор `Private :: Do Not Upload`, предусмотренный [PyPA](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/) для запрета загрузки на PyPI. Он не управляет visibility на GitHub; публичная публикация пакета пока не входит в задачу.
