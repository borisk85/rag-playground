"""
ШАГ 4 (финал): полный RAG-цикл.

  вопрос -> [поиск в ChromaDB] -> найденные куски -> [Claude] -> ответ

Ключевая идея финала: Claude отвечает НЕ из своих общих знаний, а строго по
найденным кускам (grounding). Поэтому ответ опирается на твой документ, а не на
то, что модель "где-то слышала". Это и есть весь смысл RAG.
"""

import sys
from pathlib import Path

from rag_core import load_documents, chunk_text, index_chunks, answer

sys.stdout.reconfigure(encoding="utf-8")

DATA_DIR = Path(__file__).parent / "data" / "raw"


if __name__ == "__main__":
    # Готовим базу (один раз заливается на диск).
    chunks = chunk_text(load_documents(DATA_DIR), chunk_size=800, overlap=100)
    index_chunks(chunks)

    question = "Какие модули доступны в бесплатном тарифе VELA и что они умеют?"
    print(f"ВОПРОС: {question}\n")

    result = answer(question, n_results=3)

    print("ОТВЕТ CLAUDE (построен только по найденным кускам):")
    print(result["answer"])
    print()

    print("--- НА ЧЁМ ОСНОВАН ОТВЕТ (источники) ---")
    for i, src in enumerate(result["sources"]):
        print(f"[{i + 1}] (distance={src['distance']:.3f}) {src['text'][:150]}...")
