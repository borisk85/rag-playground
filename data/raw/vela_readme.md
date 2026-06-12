# VELA — AI Agent Builder SaaS

Платформа для создания персональных ИИ-ботов в Telegram. Под капотом Claude от Anthropic. Без кода, без серверов, за 5 минут.

🌐 [velabot.io](https://velabot.io)

## Что это

VELA — SaaS-платформа: пользователь подключает свой Telegram-бот через BotFather, выбирает модули (погода, Google Calendar, Gmail, напоминания и т.д.) — и получает персонального ИИ-ассистента в Telegram.

Ключевые принципы:
- **Multi-tenant webhook** — один FastAPI сервер обслуживает тысячи ботов клиентов через `POST /webhook/{token_hash}`
- **Платформенные API ключи** — клиент не знает про Anthropic, не имеет API ключей. Платформа использует свой ключ для всех ботов.
- **Изоляция данных** — каждый бот видит только свою историю и настройки в Redis
- **Безопасность** — токены ботов хранятся зашифрованными (Fernet)

## Структура монорепо

```
agent-builder-saas/
├── src/                # Backend: FastAPI + python-telegram-bot + Claude
│   ├── main.py
│   ├── api/            # Webhook, OAuth, dashboard, billing
│   ├── bot/            # Dispatcher и handlers Telegram-сообщений
│   ├── modules/        # Модули агента (погода, Gmail, calendar, и т.д.)
│   ├── ai/             # Claude client с tool_use loop
│   ├── auth/           # Google OAuth
│   ├── storage/        # Redis + PostgreSQL
│   └── billing/        # Lemon Squeezy
│
├── landing/            # Frontend: Next.js 16 (лендинг + дашборд + блог + админка)
│   ├── app/            # App Router
│   ├── components/     # React-компоненты
│   ├── lib/            # blog-posts.ts и общая логика
│   └── public/
│
├── tests/              # Pytest
├── requirements.txt    # Python зависимости (бэкенд)
└── railway.toml        # Конфиг Railway
```

## Технический стек

| Компонент | Технология |
|---|---|
| Web + Webhook | FastAPI + uvicorn |
| Telegram | python-telegram-bot (webhook mode) |
| AI | Claude Sonnet 4.6 + Haiku 4.5 |
| Сессии | Redis |
| БД | PostgreSQL |
| Шифрование | cryptography (Fernet) |
| Биллинг | Lemon Squeezy |
| Email | Brevo |
| Auth | Clerk |
| Frontend | Next.js 16 + React + TypeScript + Tailwind |
| Hosting backend | Railway |
| Hosting frontend | Vercel |

## Деплой

### Backend (Railway)
Автоматически из main ветки GitHub. Railway сам пересобирает контейнер при push.

### Frontend (Vercel)
Вручную из папки `landing/`:
```bash
cd "landing" && npx vercel --prod
```
**ВАЖНО:** git push не деплоит лендинг автоматически — нужен ручной vercel CLI.

## Тарифы и план

| План | Цена | Сообщ./день | Память | Модули |
|---|---|---|---|---|
| Free | $0 | 30 | 30 сообщ. / 7 дней | погода, курсы, напоминания, поиск, анализ фото и документов |
| Starter | $5/мес | 75 | 50 сообщ. | + поиск авиабилетов, ценовые уведомления, расширенный крипто, индексы/сырье |
| Professional | $10/мес | безлимит | 100 сообщ. + долгосрочная память | + Google Workspace (Calendar/Gmail/Tasks/Drive) + генерация картинок |
| Creator | $29/мес | безлимит | 200 / безлимит | + генерация картинок + до 3 ботов |

## Связанные репозитории

- [borisk85/vela-marketing-bot](https://github.com/borisk85/vela-marketing-bot) — внутренний бот для генерации SEO-статей в блог velabot.io/blog. Claude Sonnet с фактчеком по `knowledge_base.md`, открывает PR в этот репо.
- [borisk85/support-bot](https://github.com/borisk85/support-bot) — внутренний бот для приёма уведомлений (баг-репорты, оплаты, NPS) и ответа пользователям через Brevo email + Telegram-нудж через их собственный бот.

## Разработка

См. `CLAUDE.md` в корне репо — там полные инструкции по структуре, командам, переменным окружения и правилам кода.
