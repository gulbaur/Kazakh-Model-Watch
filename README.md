# Kazakh Model Watch

Небольшой benchmark AI-моделей на казахском языке с вручную проверенными заданиями и сохранёнными ответами. Приоритет MVP — минимальный работающий pipeline и реальные измерения.

**Статус:** Dataset Loader, Scoring Engine, Model Adapter Foundation, OpenRouter Integration и offline foundation Evaluation Runner. Есть CLI `validate` / `run` / `report`, локальные артефакты и synthetic mock-режим. Все 24 задания остаются черновиками; две реальные модели в конфигурации имеют статус `not_ready`. 2026-10-15 выполнен один подтверждённый smoke-вызов `qwen/qwen3.5-9b`, получен ответ `5`. Offline workflow GitHub Actions реализован; базовая интеграция проверена CI на Python 3.11 и 3.13. Новых live-вызовов и полного реального сравнения на этапе runner не было.

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
├── .github/                 # offline CI, шаблоны issue/PR и safety check
├── config/
│   ├── openrouter_model.json # исходный публичный снимок для smoke CLI
│   ├── models.json           # два реальных кандидата, пока not_ready
│   └── mock_models.json      # два явно синтетических offline mock
├── data/
│   ├── benchmark.json       # пустая исходная заготовка; не используется loader
│   └── benchmark.jsonl      # 24 отобранных кандидата, review.status=draft
├── docs/
│   └── MVP_SPEC.md          # основная техническая спецификация
├── src/
│   ├── .gitkeep
│   └── kazakh_model_watch/
│       ├── __init__.py
│       ├── __main__.py      # CLI validate / run / report
│       ├── config.py        # offline-валидация публичной конфигурации
│       ├── dataset.py       # загрузка и валидация JSONL
│       ├── model_adapter.py # ModelRequest, ModelResponse, интерфейс и offline mock
│       ├── normalization.py # общая нормализация kk-short-v1
│       ├── openrouter_adapter.py
│       ├── openrouter_smoke.py
│       ├── runner.py        # preflight, последовательные вызовы, retry
│       ├── reporting.py     # атомарные артефакты и offline report
│       └── scoring.py       # exact match и normalized correctness
├── tests/
│   ├── .gitkeep
│   ├── conftest.py
│   ├── test_cli.py
│   ├── test_config.py
│   ├── test_dataset.py
│   ├── test_model_adapter.py
│   ├── test_openrouter_adapter.py
│   ├── test_reporting.py
│   ├── test_runner.py
│   └── test_scoring.py
├── .gitignore
├── AGENTS.md
├── LICENSE
├── README.md
└── pyproject.toml
```

`data/benchmark.jsonl` содержит черновики для разработки: два задания numbers, два dates и одно morphology. Эталоны — кандидаты для будущей ручной проверки. Публичные конфигурации находятся в `config/`; артефакты запуска создаются в игнорируемом `results/<run_id>/`. Корневой `LEADERBOARD.md` на этом этапе не формируется.

## Локальное окружение

Python **3.11+**. Обычный `venv` и `pip` достаточны; отдельный менеджер окружения не требуется. `pyproject.toml` фиксирует прямые зависимости: setuptools, pytest, HTTPX для HTTP и python-dotenv для явной локальной загрузки ключа. SDK провайдера не используется.

Из корня репозитория, PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]" pytest-socket==0.7.0
```

Linux/macOS:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]" pytest-socket==0.7.0
```

Активация окружения не обязательна: команды используют его интерпретатор напрямую. Установка добавляет пакет, зависимости и CLI через `python -m kazakh_model_watch`. Конфигурация setuptools использует существующий `src/`, тесты находятся в `tests/`.

Запуск тестов из корня:

```powershell
.\.venv\Scripts\python.exe -m pytest -q --disable-socket
```

На Linux/macOS использовать `.venv/bin/python -m pytest -q --disable-socket`. Плагин pytest-socket блокирует сетевые соединения на время тестов, как в CI. Локальная команда использует стандартную автозагрузку установленного плагина; в CI автозагрузка отключена и плагин подключается явно через `-p pytest_socket`. Тесты проверяют loader, scoring, adapters, конфигурацию, CLI, retry и локальные артефакты. Они не доказывают языковую правильность черновиков. Метаданные `verified` и capability metadata в новых runner fixtures синтетические и не подтверждают реальную ручную проверку или вызов модели.

## Evaluation Runner: offline foundation

Все команды выполняются из корня проекта. В Linux/macOS заменить Windows-путь на `.venv/bin/python`.

Offline-проверка чернового датасета и конфигурации, без ключа, чтения `.env` и запросов metadata:

```powershell
.\.venv\Scripts\python.exe -m kazakh_model_watch validate --development
```

Без `--development` команда `validate` проверяет финальные 24 verified задания (по 8 на категорию) и готовность обеих моделей. Текущие 24 draft эту проверку не проходят. Валидация структуры не устанавливает факт человеческой проверки.

Локальная разработка на двух mock-моделях:

```powershell
.\.venv\Scripts\python.exe -m kazakh_model_watch run --mock --config config/mock_models.json --run-id offline-demo-001
.\.venv\Scripts\python.exe -m kazakh_model_watch report results/offline-demo-001
```

Это синтетические ответы для проверки pipeline; mock не получает эталоны и не подставляет правильные ответы. Его usage по умолчанию неизвестен, latency и elapsed остаются `null`. Повторное использование `run_id` запрещено: выбрать новый ID либо опустить `--run-id` для автоматической генерации. CLI не перезаписывает прежние запуски и не возобновляет генерацию; `report` только пересчитывает существующий запуск offline, включая неполный, без ключей и адаптеров.

Для реального `run` нужен явный `--live` и отдельное разрешение владельца на расходы. До конструирования адаптеров проверяются финальный датасет, весь конфиг, готовность двух моделей, общий профиль и все ENV-ключи. Без opt-in сетевых вызовов нет. `.env` runner не загружает; ключ берётся только из `OPENROUTER_API_KEY`. В этой issue реальный `run` не выполнялся.

В [config/models.json](config/models.json) отдельно настроены `qwen/qwen3.5-9b` и `google/gemini-3.1-flash-lite`. Публичные Models/Endpoints metadata проверены 2026-10-19 и сохранены без тарифов и ключей. Поддержка имён temperature/max_tokens объявлена, но точная совместимость лимита **видимого** ответа не подтверждена; у Gemini metadata указывают включённый по умолчанию reasoning. Поэтому оба кандидата остаются `not_ready` с причинами и `profile_verified=false`. Этот флаг — поле локального согласования, не свидетельство API; не менять его без review. Неподдерживаемые настройки не заменяются автоматически. Профиль сохранён: temperature 0, max output tokens 128, timeout 60, stream false, один user message с общей инструкцией.

Issue #9 добавляет в общий профиль **предлагаемое** `reasoning: {"enabled": false}` и проводит его через config → ModelRequest → OpenRouterAdapter → runner/manifest. Старые конфигурации без поля сохраняют прежний payload; mock работает offline. Разрешён только этот вариант reasoning, без effort/exclude и дополнительных настроек. Сообщённый reasoning при запрошенном отключении даёт `api_error/reasoning_not_disabled` с сохранением видимого текста и usage; `length` остаётся `truncated`. Ни одна модель не переведена в ready. Ограничения upstream, оставшееся расхождение с `mvp-v1` и предлагаемый smoke с отдельным бюджетом описаны в [docs/ISSUE9_NON_THINKING_PROFILE.md](docs/ISSUE9_NON_THINKING_PROFILE.md). Перед реальным сравнением нужны согласование новой версии протокола и фактическая проверка; MVP_SPEC.md не изменён.

Порядок заданий перемешивается локальным `Random(42)`, порядок моделей циклически чередуется. Runner делает максимум две попытки: повтор только после timeout, network error или HTTP 429/5xx. Ожидание — максимум из 2 секунд и Retry-After; при Retry-After больше 60 секунд повтора нет. Поддерживается Retry-After в секундах и в формате HTTP-date; сохраняются разобранная задержка, история реальных попыток, запланированный `retry_delay_s` и факт завершения ожидания `retry_wait_completed`. Качество ответа, 401/402, content_filter, empty_response и truncated не вызывают повторов. Внутренних retries/fallback у адаптера нет.

В `results/<run_id>/` сохраняются `manifest.json`, точные входные `dataset.jsonl` и `config.json`, `responses.json/csv`, `metrics.json/csv` и `leaderboard.md`. Manifest фиксирует версии протокола/датасета/scoring/Unicode, SHA256 входов и инструкции, Git SHA/dirty, окружение, профиль и фактический порядок. Неизвестные версии зависимостей остаются `null`, включая build-only setuptools при изолированной сборке. JSON заменяется атомарно после каждой попытки; при прерывании manifest остаётся `incomplete`, уже полученные ответы сохраняются, отсутствующие пары имеют `not_run` и неизвестное время. Report проверяет snapshots, идентичность пар и retry-цепочки и пересчитывает scoring по целому ответу.

Accuracy, exact match и coverage используют полный знаменатель; сохраняются статусы и разбивка по категориям. Latency реального вызова измеряется монотонным таймером; среднее/медиана берутся только по `ok`, включая неверные ответы. Usage хранится по каждой попытке, отсутствующие счётчики не вычисляются. Стоимость не собирается: `null` / `N/A` с причиной. Файл `leaderboard.md` внутри mock-каталога явно помечен **OFFLINE SYNTHETIC**, неполный запуск — **incomplete**. Публикация корневого leaderboard и реальный benchmark не входят в текущий этап. Риск лимита токенов из [Task 5 review](docs/TASK5_REVIEW.md) остаётся открытым.

## Dataset Loader

```python
from kazakh_model_watch.dataset import load_dataset

# Разработка: любой непустой набор, черновики разрешены.
drafts = load_dataset("data/benchmark.jsonl")

# Финальная проверка: 24 verified-задания, по 8 на категорию.
# Для текущих 24 черновиков этот вызов ожидаемо завершится ошибкой.
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

Любой `status != "ok"` даёт нули по обеим оценкам, даже при совпадающем тексте. `None`, пустой и пробельный текст также дают нули. Для `status="ok"` текст другого типа вызывает `TypeError`; автоматического преобразования в строку нет. Числа из объяснений не извлекаются, Markdown и пунктуация не удаляются, похожие буквы и zero-width символы сохраняются. Исходные ответы и эталоны не изменяются. Scoring работает локально, без API; runner и offline report переиспользуют эту функцию перед агрегацией и экспортом.

## Model Adapter Foundation

`ModelRequest` и `ModelResponse` — dataclass-структуры с проверкой типов и значений при создании. Их поля сохраняют исходный prompt и видимый ответ; атрибуты нельзя переназначать. В запросе — provider, model ID, полный prompt и параметры; defaults из спецификации: temperature `0`, max output tokens `128`, timeout `60` секунд. Prompt собирается вызывающим кодом; адаптер не добавляет инструкции и не получает эталоны.

Ответ содержит восемь полей: `status`, `raw_response`, `requested_model_id`, `returned_model_id`, `request_id`, `finish_reason`, `usage`, `latency_ms`. Неизвестные metadata/usage/latency остаются `None`; нулевые счётчики и latency не заменяются на `None`. У `ok` должен быть непробельный текст, а ошибки и обрывы имеют `latency_ms=None`, поскольку успешная попытка отсутствует. `usage` содержит только неотрицательные целочисленные счётчики токенов; их значения не вычисляются из длины текста.

`ModelAdapter` задаёт синхронный метод `generate(request) -> response` и сам использует только стандартную библиотеку. Плагинов и registry нет; конкретная интеграция OpenRouter описана ниже. Таймер, ограниченные повторы, `attempts`, `error` и `total_elapsed_ms` относятся к runner.

Минимальное расширение для review issue #3 — необязательный `get_diagnostic()`, по умолчанию возвращающий `{}`. OpenRouter возвращает копию безопасных `http_status`, `error_code`, `request_attempted`, `retry_after_s`; тело ошибки и заголовки туда не попадают. Исходные восемь полей ModelResponse, `generate()` и прежний `last_diagnostic` сохранены. Без диагностики `api_error` не считается сетевым сбоем и не повторяется; известный `timeout` допускает повтор по общему правилу.

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

Вывод содержит поля `ModelResponse`, измеренный `elapsed_ms`, диагностический `http_status` и статический `error_code`. `elapsed_ms` измеряет весь вызов generate, включая HTTP и преобразование ответа; до HTTP-попытки остаётся `None`. В smoke CLI `ModelResponse.latency_ms` остаётся `None`; evaluation runner измеряет свои попытки отдельно. Код выхода `0` означает ok, `1` — ошибку, пустой или обрезанный ответ. Raw text не нормализуется и не оценивается этой командой.

HTTP ошибки, timeout, пустой/обрезанный ответ и некорректная схема обрабатываются явно. Текст исключений, тела ошибок и Authorization header не выводятся. Если сервер отражает ключ в JSON, ответ отклоняется целиком с `sensitive_response`. `usage` сохраняет только сообщённые API целочисленные счётчики токенов; вложенные token details представлены через dotted keys. Пропущенные счётчики не вычисляются, cost-поля не превращаются в стоимость или токены.

Реальная проверка 2026-10-15: ровно один подтверждённый POST, HTTP `200`, status `ok`, requested/returned model ID `qwen/qwen3.5-9b`, raw response `5`, finish reason `stop`, измеренный `elapsed_ms=7116.382699459791`. API сообщил prompt/completion/total tokens `83/293/376`, включая `291` reasoning token. Вывод сохранён локально в игнорируемом `logs/`; это проверка интеграции на одном задании, без результатов benchmark и leaderboard. Стоимость не рассчитывалась.

При переданном `max_tokens=128` API сообщил `completion_tokens=293`. Счётчики сохранены как получены; причина расхождения для выбранного backend не установлена. По [документации OpenRouter](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens#reasoning-tokens-and-max_tokens), у большинства провайдеров reasoning и видимый ответ делят этот лимит. Перед полноценным сравнением нужно проверить фактическую семантику лимита для этого endpoint. Параметры запроса и спецификация по результату smoke не изменялись, дополнительных вызовов не было.

Ограничения: снимок Models API может устареть; require_parameters проверяется при маршрутизации, но доступность и quota подтверждаются только реальным вызовом. Alias модели может изменяться, возвращённый ID сохраняется. Поддерживается один текстовый completion; tool calls, массивы content и неожиданные модели отклоняются. HTTPX timeout задаётся для сетевых фаз, отдельного общего deadline нет. Исторический smoke CLI не задаёт reasoning; новый опциональный режим требует явного поля в ModelRequest. Finish_reason=length всегда даёт truncated. Smoke CLI рассчитан на локальную editable-установку и присутствие config-файла; это не Evaluation Runner.

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
