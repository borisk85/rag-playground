"""
Ядро RAG-пайплайна. Здесь живёт переиспользуемая логика всех шагов.
Пронумерованные файлы (01_*, 02_*, ...) — это "запускалки", которые
импортируют отсюда функции и показывают результат конкретного шага.

Так делают в проде: бизнес-логика — в модуле, точки входа — отдельно.
"""

import re
from pathlib import Path

import anthropic
import chromadb
import voyageai
from dotenv import load_dotenv
from pypdf import PdfReader
from rank_bm25 import BM25Okapi

# Загружаем ключи из .env в переменные окружения (один раз при импорте модуля).
load_dotenv(Path(__file__).parent / ".env")

# Модель эмбеддингов Voyage. voyage-4-lite входит в бесплатный tier (200M токенов)
# и даёт прод-уровень качества. Сменить модель = поменять одну эту строку.
EMBED_MODEL = "voyage-4-lite"


# --------------------------------------------------------------------------
# ШАГ 1: загрузка и нарезка
# --------------------------------------------------------------------------

def load_pdf(path: Path) -> str:
    """Читает PDF и склеивает текст всех страниц в одну строку."""
    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def _strip_frontmatter(text: str) -> str:
    """Убирает YAML-шапку (--- ... ---) из markdown, если она есть. Это шум для RAG."""
    if text.startswith("---"):
        end = text.find("---", 3)
        if end != -1:
            return text[end + 3:].lstrip()
    return text


def load_documents(folder: Path) -> str:
    """
    Читает ВСЕ документы из папки (.md, .txt, .pdf) и склеивает в один текст.
    В проде база знаний — это набор файлов, а не один документ. Поэтому грузим папку.
    """
    parts = []
    for path in sorted(folder.glob("*")):
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            parts.append(load_pdf(path))
        elif suffix in (".md", ".txt"):
            parts.append(_strip_frontmatter(path.read_text(encoding="utf-8")))
    return "\n\n".join(parts)


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """
    Режет текст на куски ПО ГРАНИЦАМ смысла, а не вслепую по символам.

    Граница — пустая строка: в markdown ею разделяются абзацы, заголовки и
    таблицы. Поэтому нож не падает посреди слова или посреди таблицы.

    Логика:
      1) бьём текст на блоки по пустым строкам (абзац / таблица / заголовок = блок);
      2) склеиваем соседние блоки в кусок, пока влезает в chunk_size — так шапка
         таблицы и её строки попадают в один кусок;
      3) перекрытие (overlap): новый кусок начинаем с хвоста предыдущего, чтобы
         мысль на стыке не потерялась (хвост обрезаем по началу слова, не рвём);
      4) запасной случай: блок сам длиннее chunk_size — режем его по символам.
    """
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]

    chunks: list[str] = []
    current = ""
    for block in blocks:
        # (4) огромный блок (больше лимита) — режем по символам, как раньше.
        if len(block) > chunk_size:
            if current:
                chunks.append(current)
                current = ""
            start = 0
            step = max(1, chunk_size - overlap)
            while start < len(block):
                piece = block[start:start + chunk_size].strip()
                if piece:
                    chunks.append(piece)
                start += step
            continue

        # (2) пробуем дописать блок в текущий кусок.
        candidate = f"{current}\n\n{block}" if current else block
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            # кусок заполнен — закрываем его и начинаем новый.
            chunks.append(current)
            # (3) перекрытие: тащим хвост прошлого куска, обрезав неполное слово.
            tail = current[-overlap:]
            if " " in tail:
                tail = tail[tail.find(" ") + 1:]
            current = f"{tail}\n\n{block}" if tail else block

    if current:
        chunks.append(current)
    return chunks


# --------------------------------------------------------------------------
# ШАГ 2: эмбеддинги через Voyage
# --------------------------------------------------------------------------

# Клиент Voyage. Ключ VOYAGE_API_KEY берётся из окружения автоматически.
_voyage = voyageai.Client()


def embed_texts(texts: list[str], input_type: str) -> list[list[float]]:
    """
    Превращает список текстов в список векторов через Voyage API.

    input_type:
      "document" — когда эмбеддим куски базы (то, что ищем СРЕДИ);
      "query"    — когда эмбеддим вопрос пользователя (то, чем ищем).
    Voyage обучена различать эти два режима — это поднимает точность поиска.
    """
    result = _voyage.embed(texts, model=EMBED_MODEL, input_type=input_type)
    return result.embeddings


# --------------------------------------------------------------------------
# ШАГ 3: хранение в ChromaDB + поиск
# --------------------------------------------------------------------------

# Persistent-клиент: база сохраняется на диск в папку chroma_db (см. .gitignore).
# Это значит, что один раз залив чанки, при следующем запуске их не надо заливать снова.
_chroma = chromadb.PersistentClient(path=str(Path(__file__).parent / "chroma_db"))


def get_collection():
    """
    Коллекция = таблица в векторной БД. Тут лежат чанки и их векторы.
    hnsw:space=cosine — говорим базе мерить близость косинусным сходством
    (то самое, что мы руками считали в Шаге 2).
    """
    return _chroma.get_or_create_collection(
        name="vela_docs",
        metadata={"hnsw:space": "cosine"},
    )


def index_chunks(chunks: list[str]) -> None:
    """
    Заливает чанки в базу: эмбеддит их через Voyage и складывает в коллекцию.
    Если база уже наполнена — пропускаем (чтобы повторный запуск не дублировал).
    """
    collection = get_collection()
    if collection.count() > 0:
        return  # уже залито ранее

    # Эмбеддим все чанки как "document" (это куски базы, среди которых ищем).
    vectors = embed_texts(chunks, input_type="document")
    collection.add(
        ids=[f"chunk_{i}" for i in range(len(chunks))],  # уникальный id каждого куска
        embeddings=vectors,                              # сам вектор
        documents=chunks,                                # исходный текст (чтобы вернуть его потом)
    )


def search(query: str, n_results: int = 3) -> list[dict]:
    """
    Ищет n_results самых близких по смыслу чанков к вопросу.
    Возвращает список словарей: {text, distance}.
    distance = 1 - cosine_similarity, поэтому МЕНЬШЕ distance = БЛИЖЕ по смыслу.
    """
    collection = get_collection()

    # Вопрос эмбеддим как "query" (прод-практика из Шага 2).
    query_vector = embed_texts([query], input_type="query")[0]

    result = collection.query(query_embeddings=[query_vector], n_results=n_results)

    docs = result["documents"][0]
    distances = result["distances"][0]
    return [{"text": d, "distance": dist} for d, dist in zip(docs, distances)]


# --------------------------------------------------------------------------
# ШАГ 4: генерация ответа через Claude по найденным кускам
# --------------------------------------------------------------------------

# Клиент Anthropic. Ключ ANTHROPIC_API_KEY берётся из окружения автоматически.
_claude = anthropic.Anthropic()
GEN_MODEL = "claude-sonnet-4-6"

# Инструкция, которая ПРЕВРАЩАЕТ поиск в RAG. Тут два требования:
#  1) отвечай ТОЛЬКО по контексту (не по своим знаниям) — это "grounding";
#  2) если в контексте ответа нет — честно скажи, не выдумывай.
# Это главный приём против галлюцинаций: модель отвечает "по бумажке".
SYSTEM_PROMPT = (
    "Ты ассистент, отвечающий на вопросы СТРОГО по предоставленному контексту. "
    "Используй только факты из контекста, не добавляй свои знания. "
    "Если в контексте нет ответа — так и скажи: 'В документе нет ответа на этот вопрос'. "
    "Отвечай кратко и по делу."
)


def answer(question: str, n_results: int = 3) -> dict:
    """
    Полный RAG-цикл: ищем релевантные куски и отдаём их Claude вместе с вопросом.
    Возвращает {answer, sources} — ответ модели и куски, на которых он построен.
    """
    found = search(question, n_results=n_results)

    # Склеиваем найденные куски в один блок контекста для конверта.
    context = "\n\n---\n\n".join(item["text"] for item in found)

    user_content = f"Контекст:\n{context}\n\nВопрос: {question}"

    response = _claude.messages.create(
        model=GEN_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}],
    )

    return {"answer": response.content[0].text, "sources": found}


# --------------------------------------------------------------------------
# ШАГ 5: reranking — overfetch + переоценка cross-encoder'ом
# --------------------------------------------------------------------------

# Reranker Voyage. rerank-2.5-lite — последний lite-reranker (контекст 32K),
# то же "lite"-семейство, что и наш эмбеддер voyage-4-lite: быстрый и дешёвый.
# Сменить = поменять одну эту строку.
RERANK_MODEL = "rerank-2.5-lite"


def search_rerank(
    query: str,
    fetch_k: int = 15,
    top_k: int = 3,
    min_score: float | None = None,
) -> list[dict]:
    """
    Двухступенчатый поиск (overfetch + rerank):
      1) overfetch — быстрый векторный поиск тащит ШИРОКО fetch_k кандидатов;
      2) rerank    — cross-encoder читает каждую пару (вопрос, чанк) ВМЕСТЕ
                     и ставит точный relevance_score 0..1, переупорядочивает.

    Обрезка результата — двумя способами:
      top_k     — оставить фиксированное число лучших (как в базовом search);
      min_score — оставить ВСЕХ, у кого score >= порога. Это ДИНАМИЧЕСКИЙ режим:
                  на узкий вопрос вернётся 1 кусок, на обзорный — несколько.
    Если задан min_score, top_k игнорируется.
    """
    candidates = search(query, n_results=fetch_k)
    docs = [c["text"] for c in candidates]

    # При min_score просим reranker вернуть ВЕСЬ пул отранжированным (top_k=None),
    # порог применяем сами ниже. Иначе сразу режем по top_k на стороне Voyage.
    reranked = _voyage.rerank(
        query,
        docs,
        model=RERANK_MODEL,
        top_k=None if min_score is not None else top_k,
    )

    results = [
        {"text": r.document, "relevance_score": r.relevance_score}
        for r in reranked.results
    ]
    if min_score is not None:
        results = [r for r in results if r["relevance_score"] >= min_score]
    return results


# --------------------------------------------------------------------------
# ШАГ 6: hybrid search — смысловой (векторный) + словесный (BM25)
# --------------------------------------------------------------------------

# Векторный поиск силён в смысле (синонимы, перефраз), но слаб на точных редких
# словах (имена, коды: Fernet, token_hash). BM25 — наоборот: ловит точное слово,
# но не понимает смысл. Гибрид гоняет оба и сливает — слепые зоны взаимно гасятся.

_bm25 = None          # ленивый кэш BM25-индекса
_bm25_docs: list[str] = []


def _tokenize(text: str) -> list[str]:
    """Грубая токенизация: слова в нижнем регистре. \\w ловит и кириллицу."""
    return re.findall(r"\w+", text.lower())


def _get_bm25() -> tuple[BM25Okapi, list[str]]:
    """Строит (один раз) BM25-индекс из тех же чанков, что лежат в ChromaDB."""
    global _bm25, _bm25_docs
    if _bm25 is None:
        _bm25_docs = get_collection().get()["documents"]
        _bm25 = BM25Okapi([_tokenize(d) for d in _bm25_docs])
    return _bm25, _bm25_docs


def keyword_search(query: str, n_results: int = 15) -> list[dict]:
    """Словесный поиск (BM25): ранжирует чанки по точному совпадению слов."""
    bm25, docs = _get_bm25()
    scores = bm25.get_scores(_tokenize(query))
    order = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)
    return [{"text": docs[i], "score": float(scores[i])} for i in order[:n_results]]


def hybrid_search(query: str, n_results: int = 3, candidates: int = 15, k: int = 60) -> list[dict]:
    """
    Гибрид: смысловой + словесный поиск, слитые через RRF (reciprocal rank fusion).

    RRF: каждый список "голосует" за документ силой 1/(k + позиция_в_списке).
    Документ, стоящий высоко в ОБОИХ списках, набирает больше всех и выходит вперёд.
    k=60 — стандартное сглаживание, чтобы топ-1 одного списка не забивал всё.
    Берём из каждого поиска candidates кандидатов, сливаем, отдаём n_results лучших.
    """
    sem = search(query, n_results=candidates)
    kw = keyword_search(query, n_results=candidates)

    fused: dict[str, float] = {}
    for rank, item in enumerate(sem):
        fused[item["text"]] = fused.get(item["text"], 0.0) + 1.0 / (k + rank)
    for rank, item in enumerate(kw):
        fused[item["text"]] = fused.get(item["text"], 0.0) + 1.0 / (k + rank)

    order = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
    return [{"text": t, "rrf_score": s} for t, s in order[:n_results]]
