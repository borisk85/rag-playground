"""
ШАГ 5 цепочки RAG: overfetch + reranking.

Проблема, которую решаем: базовый векторный поиск (bi-encoder) считает вектор
вопроса и вектор чанка ПОРОЗНЬ, а потом сравнивает. Это быстро, но грубо —
вектор чанка не "видит" вопрос. Иногда самый полезный кусок оказывается не на
1-м месте.

Решение в два прохода:
  1) overfetch — дёшево тащим ШИРОКИЙ пул кандидатов (fetch_k штук);
  2) rerank    — cross-encoder читает пару (вопрос, чанк) ВМЕСТЕ и ставит точный
                 relevance_score. Дорого, поэтому гоняем только по узкому пулу.

Этот файл печатает "до/после", чтобы увидеть переупорядочивание своими глазами.
"""

import sys
from pathlib import Path

from rag_core import load_documents, chunk_text, index_chunks, search, search_rerank

sys.stdout.reconfigure(encoding="utf-8")

DATA_DIR = Path(__file__).parent / "data" / "raw"
FETCH_K = 15  # размер overfetch-пула (у нас 49 чанков, 15 ≈ 30% базы — дёшево)


if __name__ == "__main__":
    chunks = chunk_text(load_documents(DATA_DIR), chunk_size=800, overlap=100)
    index_chunks(chunks)
    print(f"В базе {len(chunks)} чанков. Overfetch={FETCH_K}.\n")

    question = "Какие интеграции и внешние инструменты поддерживает бот?"
    print(f"ВОПРОС: {question}\n")

    # --- ДО: базовый поиск, первые 3 из пула (что ушло бы в Claude без rerank) ---
    base = search(question, n_results=FETCH_K)
    print("=== ДО reranking (top-3 по distance, меньше = ближе) ===")
    for i, item in enumerate(base[:3]):
        print(f"#{i + 1}  distance={item['distance']:.4f}  | {item['text'][:90]}...")

    # --- ПОСЛЕ: тот же пул, переоценённый cross-encoder'ом ---
    ranked = search_rerank(question, fetch_k=FETCH_K, top_k=3)
    print("\n=== ПОСЛЕ reranking (top-3 по relevance_score, больше = релевантнее) ===")
    for i, item in enumerate(ranked):
        print(f"#{i + 1}  score={item['relevance_score']:.4f}  | {item['text'][:90]}...")

    # --- Весь пул по score: по нему видно, где провести порог min_score ---
    full = search_rerank(question, fetch_k=FETCH_K, min_score=0.0)
    print(f"\n=== Все {len(full)} кандидата по relevance_score (для выбора порога) ===")
    for item in full:
        print(f"  {item['relevance_score']:.4f}  | {item['text'][:70]}...")
