"""
ШАГ 6 цепочки RAG: измеряем качество поиска ЧИСЛОМ, а не на глаз.

Идея: набор контрольных вопросов, для каждого известно слово-маркер, которое
ОБЯЗАНО быть в правильном куске. Метрика Hit@K — в скольких вопросах нужный
кусок попал в топ-K. Поменял систему → прогнал снова → видно, число выросло
или упало. Здесь же сравниваем два поиска: только смысловой против гибрида.
"""

import sys
from pathlib import Path

from rag_core import load_documents, chunk_text, index_chunks, search, hybrid_search

sys.stdout.reconfigure(encoding="utf-8")
DATA = Path(__file__).parent / "data" / "raw"

index_chunks(chunk_text(load_documents(DATA), 800, 100))

# (вопрос, слово-маркер правильного куска). Маркеры — точные редкие слова:
# на них видно разницу между смысловым и словесным поиском.
TESTS = [
    ("Чем шифруются токены ботов?", "Fernet"),
    ("Какой сервис используется для биллинга?", "Lemon Squeezy"),
    ("На каком хостинге работает бэкенд?", "Railway"),
    ("Какая база данных хранит данные?", "PostgreSQL"),
    ("Как деплоится фронтенд?", "vercel"),
    ("Какие модели Claude использует платформа?", "Sonnet 4.6"),
    ("По какому адресу приходят сообщения ботов?", "/webhook/{token_hash}"),
    ("Сколько сообщений в день на бесплатном тарифе?", "30 сообщ. / 7 дней"),
]
K = 3


def hit(results: list[dict], needle: str) -> bool:
    return any(needle.lower() in r["text"].lower() for r in results[:K])


sem_hits = hyb_hits = 0
print(f"Hit@{K}: попал ли кусок с правильным ответом в топ-{K}.\n")
for q, needle in TESTS:
    sh = hit(search(q, n_results=K), needle)
    hh = hit(hybrid_search(q, n_results=K), needle)
    sem_hits += sh
    hyb_hits += hh
    m = lambda b: "OK  " if b else "MISS"
    print(f"  смысл:{m(sh)}  гибрид:{m(hh)}  | {q}")

n = len(TESTS)
print(f"\nИТОГ Hit@{K}:")
print(f"  только смысловой поиск : {sem_hits}/{n}")
print(f"  гибрид (смысл + слова) : {hyb_hits}/{n}")
