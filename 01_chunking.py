"""
ШАГ 1 цепочки RAG: загрузка документа + нарезка на чанки (chunking).

Зачем: LLM нельзя скормить весь документ на каждый вопрос — дорого и модель
тонет в лишнем. Поэтому режем документ на куски, чтобы позже находить только
релевантные. Здесь нет ни эмбеддингов, ни базы, ни LLM — голый фундамент.

Логика load_pdf / chunk_text живёт в rag_core.py. Этот файл — запускалка,
которая показывает результат шага глазами.
"""

import sys
from pathlib import Path

from rag_core import load_documents, chunk_text

# Windows-консоль по умолчанию не UTF-8 и спотыкается на кириллице при print().
sys.stdout.reconfigure(encoding="utf-8")

DATA_DIR = Path(__file__).parent / "data" / "raw"


if __name__ == "__main__":
    text = load_documents(DATA_DIR)
    print(f"Загружен документ: {len(text)} символов\n")

    chunks = chunk_text(text, chunk_size=800, overlap=100)
    print(f"Получилось чанков: {len(chunks)}\n")

    for i, chunk in enumerate(chunks[:3]):
        print(f"--- ЧАНК {i} ({len(chunk)} символов) ---")
        print(chunk)
        print()
