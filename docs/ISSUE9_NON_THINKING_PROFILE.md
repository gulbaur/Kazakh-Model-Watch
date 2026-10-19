# Issue #9: предлагаемый non-thinking профиль

Дата проверки документации: 2026-10-19. Статус: offline implementation, без live-подтверждения.

## Что подтверждено источниками

OpenRouter использует единый объект `reasoning`; запрос отключения — `{"reasoning":{"enabled":false}}`. `reasoning.exclude=true` только скрывает reasoning из ответа, не отключая его генерацию или оплату. Возможности различаются между моделями и провайдерами; metadata `reasoning.mandatory=true` означают, что отключение недопустимо. На большинстве провайдеров `max_tokens` охватывает reasoning и видимый ответ вместе. Источник: [Reasoning Tokens](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens).

`provider.require_parameters=true` ограничивает маршрутизацию провайдерами, объявляющими поддержку запрошенных параметров. Это проверка заявленных возможностей, не измерение поведения конкретного backend. Источник: [Provider Selection](https://openrouter.ai/docs/guides/routing/provider-selection#requiring-providers-to-support-all-parameters).

Сохранённый публичный снимок `config/models.json` от 2026-10-19 содержит `reasoning` в supported_parameters обеих моделей и их endpoints; model-level mandatory=false. У Gemini в снимке default_enabled=true. Снимок не обновлялся догадками и не подтверждает исполнение enabled=false. Источники снимка указаны в metadata: Models API и отдельные Endpoints API для точных model IDs.

## Предложение и граница протокола

Для обеих моделей в общем `profile` добавлено только `reasoning: {"enabled": false}`. Temperature=0, max_output_tokens=128, timeout_s=60, stream=false, общий prompt, один user message, точные model IDs и запрет fallback сохранены. Настройки effort, exclude, thinking_budget и другие upstream-поля не добавляются.

Это **предлагаемое изменение общего профиля**, а не подтверждённый режим `mvp-v1`. Спецификация сейчас запрещает дополнительный reasoning-режим и определяет лимит видимого текста. `docs/MVP_SPEC.md` и поле protocol_version не изменены; старое значение версии оставлено для совместимости существующих offline-компонентов. Наличие reasoning видно в config snapshot, его SHA256 и manifest.profile; manifest также помечает профиль как предложение issue #9. Эти данные нельзя публиковать как одобренный benchmark `mvp-v1`.

До любого оценочного запуска требуется отдельно согласовать редакцию спецификации и новую версию протокола для изменённого профиля. В текущем config обе модели остаются not_ready, profile_verified=false. Зелёные тесты не снимают эти блокировки. Датасет остаётся draft; никакой человеческой проверки эта issue не устанавливает.

Старый config без reasoning остаётся допустимым; ModelRequest.reasoning по умолчанию None, поле HTTP payload отсутствует. Отдельный config/mock_models.json и исторический smoke config не менялись. Mock принимает дополнительное поле без сети; его ответы остаются синтетическими.

## Проверяемое поведение реализации

- Config и ModelRequest принимают только отсутствующее поле либо объект с единственным enabled=false; в config явный null отклоняется. True, 0, строки, effort, exclude и дополнительные поля запрещены.
- Для ready OpenRouter-конфига проверяется reasoning в model и endpoint supported_parameters; обязательный reasoning отклоняется. Adapter повторно проверяет model-level возможности до ключа и HTTP. Require_parameters=true и allow_fallbacks=false передаются всегда; неподдерживаемый backend может вернуть ошибку API. Никакого запроса без reasoning вместо отклонённого нет.
- Один adapter.generate делает максимум один POST. Ограниченные инфраструктурные retries runner не менялись; HTTP 400, профильные ошибки и truncated не повторяются. Предлагаемый smoke ниже не использует runner.
- При явно запрошенном отключении, finish_reason=stop/None и сообщённых положительных reasoning_tokens либо непустом reasoning/reasoning_content/reasoning_details адаптер возвращает api_error/reasoning_not_disabled. Видимый raw_response, finish_reason, ID и сообщённый usage сохраняются; текст reasoning не копируется в видимый ответ. Scoring получает ноль.
- Content_filter сохраняет api_error/content_filtered; length всегда truncated, включая пустой content и положительные reasoning counters. Эти причины имеют приоритет перед проверкой reasoning.
- При stop и отсутствии явных свидетельств reasoning сохраняется обычная классификация ответа. Неизвестные counters не превращаются в нули. Даже usage.reasoning_tokens=0 или status=ok не доказывают свойства backend.
- Completion_tokens выше 128 сами по себе не заменяют stop на length, не исправляются и не трактуются как доказанный перерасход видимого бюджета. Это требует отдельной проверки семантики и биллинга. Прежний режим без reasoning сохраняет обработку usage.

## Предлагаемый smoke после отдельного разрешения

Этот раздел — план; команды live не выполнялись и новая smoke-команда в issue не добавляется. Старый openrouter_smoke проверяет только исторический профиль Qwen, поэтому им нельзя подтвердить оба новых профиля.

1. Согласовать с владельцем **не более двух POST**: по одному qwen/qwen3.5-9b и google/gemini-3.1-flash-lite, без retries/fallback и без benchmark. Если первый запрос выявит несоответствие, остановиться и не тратить второй автоматически.
2. В обоих запросах использовать общий профиль с reasoning.enabled=false и одно задание: «Екіге үшті қосқанда қанша болады? Жауапты тек бүтін санмен жазыңыз.» с общей инструкцией. Gold/review не передавать; .env загружать только при явном opt-in. Вызывать adapter напрямую, без retry runner.
3. Предлагаемый **верхний предел разрешённых расходов: USD 0.10 суммарно на оба запроса**, включая возможную платную неуспешную попытку. Это предложение владельцу, не расчёт тарифа и не гарантия API. Перед разрешением проверить актуальные официальные тарифы, входной размер и доступный механизм ограничения расходов. Если финансовый предел нельзя обосновать или обеспечить, запросы не выполнять. Max_tokens=128 и timeout не являются финансовым hard cap. Сейчас достоверная верхняя оценка фактического списания не установлена.
4. Локально сохранить requested/returned model ID, request ID, finish_reason, status, visible raw_response, измеренное время, HTTP status и usage как получены, плюс доступную идентичность endpoint/подтверждённый биллинг. Не выводить ключи и error bodies; logs/results не публиковать. Отсутствующие сведения оставить null.
5. Проверить отсутствие сообщённого reasoning, truncation и других несоответствий, отдельно рассмотреть output cap и реальные начисления. Один удачный ответ — свидетельство конкретного запроса, не гарантия всех upstream. При неизвестном usage вывод об отключении останется непроверенным. Решение о profile_verified/readiness и протоколе принимает владелец после review; автоматического переключения нет.
