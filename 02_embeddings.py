"""
ШАГ 2 цепочки RAG: эмбеддинги (превращение текста в векторы смысла).

Что показываем:
  A) Как выглядит вектор (размерность, первые числа).
  B) Главное свойство: похожие по смыслу тексты дают близкие векторы,
     а непохожие — далёкие. Меряем "близость" косинусным сходством.

Косинусное сходство (cosine similarity) — число от -1 до 1:
  ~1.0  -> смысл почти одинаковый
  ~0.0  -> тексты про разное
Именно по этому числу RAG потом решает, какие чанки релевантны вопросу.
"""

import sys

from rag_core import embed_texts

sys.stdout.reconfigure(encoding="utf-8")


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Косинусное сходство двух векторов. Без numpy — чтобы видеть формулу."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    return dot / (norm_a * norm_b)


if __name__ == "__main__":
    # A) Эмбеддим один кусочек и смотрим, что такое вектор.
    sample = "AI Engineer строит RAG-пайплайны и агентов поверх готовых моделей."
    vector = embed_texts([sample], input_type="document")[0]
    print(f"Текст превращён в вектор из {len(vector)} чисел.")
    print(f"Первые 5 чисел: {[round(x, 4) for x in vector[:5]]}\n")

    # B) Проверяем главное свойство на трёх текстах.
    query = "Чем занимается инженер по искусственному интеллекту?"
    close = "AI Engineer создаёт production-системы на базе LLM."
    far = "Рецепт борща: свёкла, капуста, картофель и мясной бульон."

    # Вопрос эмбеддим как query, остальное — как document (прод-практика).
    q_vec = embed_texts([query], input_type="query")[0]
    close_vec, far_vec = embed_texts([close, far], input_type="document")

    print(f"Вопрос: {query}\n")
    print(f"Сходство с близким по смыслу ('{close}'):")
    print(f"  -> {cosine_similarity(q_vec, close_vec):.4f}  (должно быть высоким)\n")
    print(f"Сходство с далёким по смыслу ('{far}'):")
    print(f"  -> {cosine_similarity(q_vec, far_vec):.4f}  (должно быть низким)")
