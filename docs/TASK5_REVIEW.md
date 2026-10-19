# Task 5 — review fixes

Проверено 2026-10-15. Новых inference-запросов не выполнялось. Использованы локальные файлы, официальная документация и два публичных GET без ключа: Models API и Endpoints API. Содержимое `.env` не открывалось; параметры генерации, исходный снимок `config/openrouter_model.json` и `MVP_SPEC.md` не изменялись.

## Исправление finish_reason

`content_filter` теперь возвращает `api_error` с диагностическим кодом `content_filtered` при любом содержимом ответа. Исходные `raw_response`, `finish_reason`, ID и usage сохраняются. В существующем enum нет отдельного статуса фильтрации; `truncated` относится к лимиту токенов. Полный текстовый отказ при `stop` остаётся `ok` и оценивается scorer как неверный ответ согласно [MVP_SPEC.md, разделы 4–5](MVP_SPEC.md).

Проверена обработка значений из [контракта OpenRouter](https://openrouter.ai/docs/api/reference/overview#finish-reason):

| finish_reason | Непробельный текст | Нет видимого текста |
| --- | --- | --- |
| `stop` | `ok` | `empty_response` |
| `length` | `truncated` | `truncated` |
| `content_filter` | `api_error` / `content_filtered` | `api_error` / `content_filtered` |
| `tool_calls`, `error`, неизвестная строка | `api_error` / `unexpected_finish_reason` | `api_error` / `unexpected_finish_reason` |
| Отсутствует или `null` | `ok`, finish_reason остаётся `None` | `empty_response`, finish_reason остаётся `None` |

Отсутствующие metadata допустимы по спецификации; причина завершения не подставляется. Некорректный тип либо пустая строка причины дают `api_error` / `malformed_response`. Таблица предполагает корректные остальные поля и HTTP 2xx без ошибки провайдера.

Offline-тесты дополнены случаями фильтрации с ответом `5`, пустым/пробельным текстом и `None`; проверяется сохранение диагностики и нулевой scoring. Проверены документированные причины ошибки, неизвестная причина и полный текстовый отказ.

## Подтверждённые факты о токенах

- Локальный сохранённый вывод `logs/openrouter-smoke-2026-10-15-xpNg69Rmxz8aIr9lPQQk.json`: HTTP 200, `stop`, ответ `5`, prompt/completion/total = `83/293/376`, reasoning = `291`. Код и offline-тест структуры запроса подтверждают отправку `max_tokens=128`, temperature 0, stream false, require_parameters true, allow_fallbacks false. Отдельного сырого HTTP-транскрипта нет.
- По [документации reasoning](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens#reasoning-tokens-and-max_tokens), completion включает reasoning. Разность `293 - 291 = 2` — вычисленная диагностическая оценка видимых токенов по описанному OpenRouter правилу; дополнительный API-счётчик в usage не создавался. Одна цифра в строке не доказывает количество токенов tokenizer.
- Спецификация задаёт 128 для **видимого текста**. У большинства провайдеров OpenRouter `max_tokens` ограничивает reasoning и видимый ответ вместе; `max_completion_tokens` имеет ту же оговорку. Поэтому одно превышение общего completion над 128 не доказывает нарушение требования к видимому тексту и не доказывает соблюдение лимита на стороне backend. См. [параметры OpenRouter](https://openrouter.ai/docs/api/reference/parameters#max-tokens).
- Текущий [Models API](https://openrouter.ai/api/v1/models) подтверждает точный ID `qwen/qwen3.5-9b`, canonical slug `qwen/qwen3.5-9b-20260310`, поддержку `temperature`, `max_tokens`, `reasoning`, `include_reasoning`; `top_provider.max_completion_tokens=32768`. Исходный локальный снимок от 2026-10-14 согласуется с этими полями. Новое наблюдение: `reasoning.mandatory=false`; `default_enabled`, `supported_efforts` и `supports_max_tokens` в reasoning metadata не сообщены.
- Публичный [Endpoints API](https://openrouter.ai/api/v1/models/qwen/qwen3.5-9b/endpoints) на дату проверки перечисляет Darkbloom, SiliconFlow, DeepInfra, Venice, Parasail и Together; все шесть объявляют поддержку `temperature`, `max_tokens`, `reasoning`. Это текущие объявления возможностей, не история маршрутизации smoke-запроса. [require_parameters](https://openrouter.ai/docs/guides/routing/provider-selection#requiring-providers-to-support-all-parameters) фильтрует маршруты по поддержке параметров; этот флаг сам по себе не доказывает фактическое применение лимита upstream.
- [Официальная карточка Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B#non-thinking-mode) описывает thinking по умолчанию и возможность non-thinking режима. Это согласуется с полученными reasoning tokens. Точные defaults выбранного OpenRouter endpoint из локального результата неизвестны.

## Возможные причины; не подтверждены для этого запроса

1. Backend применяет лимит к видимому ответу отдельно от thinking budget. Тогда 291 reasoning token и около двух видимых токенов совместимы с лимитом видимого текста 128. Документация оговаривает различия провайдеров, но не подтверждает такую семантику для фактически использованного endpoint.
2. Gateway/backend иначе преобразовал, ограничил или проигнорировал бюджет; возможна ошибка usage. Поддержка параметра в metadata не исключает дефект реализации. Наличие reasoning объясняет состав 293 токенов, но само по себе не объясняет превышение совокупного бюджета 128.

В сохранённом результате нет upstream provider, `native_finish_reason`, полного сырого тела ответа и effective upstream parameters. `top_provider` в Models API не устанавливает провайдера конкретной генерации. Точную причину по имеющимся свидетельствам установить нельзя.

## Рекомендуемое решение

1. Сверить существующий request ID `gen-1792047287-xpNg69Rmxz8aIr9lPQQk` с OpenRouter Activity/деталями генерации: backend, native finish reason и применённые параметры, если доступны; при необходимости передать вопрос поддержке. В этой задаче аутентифицированные запросы статистики не выполнялись.
2. До финального benchmark считать семантику бюджета Qwen неподтверждённой. Сохранять исходные usage и статусы; не обрезать ответ локально, не подменять счётчики, не считать стоимость по догадке. Общая usage выше 128 сама по себе не является `finish_reason=length` и не должна автоматически превращать ответ в `truncated`.
3. После отдельного согласования рассмотреть non-thinking профиль (`reasoning.enabled=false`) у endpoint с подтверждённой поддержкой либо выбрать endpoint/модель с документированной совместимостью исходному профилю. `mandatory=false` делает отключение кандидатом для проверки, но не подтверждает его выполнение каждым backend. Это изменение запроса и решение протокола; сейчас оно не внесено. `reasoning.exclude=true` лишь скрывает reasoning и не отключает его; замена на `max_completion_tokens` также не гарантирует отдельный бюджет видимого ответа.

Полный pytest после исправления: **381 passed in 2.01s**, ошибок нет. Лог: `logs/pytest-task5-review-2026-10-15-2e1b37.log`. Commit не создавался, benchmark не запускался.
