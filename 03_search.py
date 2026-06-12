"""
ШАГ 3 цепочки RAG: хранение чанков в ChromaDB + поиск по смыслу.

Что происходит:
  1. Режем документ на чанки (Шаг 1).
  2. Заливаем их в ChromaDB — она сама хранит векторы и умеет искать ближайшие.
  3. Задаём вопрос обычными словами — база возвращает топ-3 куска по смыслу.

Это первый "настоящий поиск": мы НЕ ищем по совпадению слов. Вопрос может быть
сформулирован иначе, чем текст в документе, — найдётся всё равно нужное.
"""

import sys
from pathlib import Path

from rag_core import load_documents, chunk_text, index_chunks, search

sys.stdout.reconfigure(encoding="utf-8")

DATA_DIR = Path(__file__).parent / "data" / "raw"


if __name__ == "__main__":
    # 1-2. Готовим базу: чанки -> ChromaDB (заливается один раз, потом берётся с диска).
    chunks = chunk_text(load_documents(DATA_DIR), chunk_size=800, overlap=100)
    index_chunks(chunks)
    print(f"В базе {len(chunks)} чанков. Ищем.\n")

    # 3. Задаём вопрос обычными словами.
    question = "Как работает напоминание и можно ли поставить его за 15 минут до встречи?"
    print(f"ВОПРОС: {question}\n")

    found = search(question, n_results=3)
    for i, item in enumerate(found):
        # distance: меньше = ближе по смыслу.
        print(f"--- НАЙДЕНО #{i + 1}  (distance={item['distance']:.4f}) ---")
        print(item["text"][:400])
        print()
