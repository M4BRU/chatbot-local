"""RAGAS evaluation — store-now, eval-on-demand."""

import asyncio
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

EVAL_ENABLED     = os.environ.get("EVAL_ENABLED", "true").lower() == "true"
EVAL_OLLAMA_URL  = os.environ.get("EVAL_OLLAMA_URL", "http://ollama-contextual:11434")
EVAL_JUDGE_MODEL = os.environ.get("EVAL_JUDGE_MODEL", "qwen3.5:0.8b")

_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ragas-batch")
_ragas_metrics: list | None = None
_batch_running = False


def _get_ragas_metrics() -> list:
    """Lazy singleton — initialise RAGAS metrics once per process."""
    global _ragas_metrics
    if _ragas_metrics is not None:
        return _ragas_metrics

    try:
        from langchain_ollama import ChatOllama
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.metrics import (
            Faithfulness,
            AnswerRelevancy,
            LLMContextPrecisionWithoutReference,
        )
        from core.embeddings import get_embeddings

        llm = LangchainLLMWrapper(
            ChatOllama(
                model=EVAL_JUDGE_MODEL,
                base_url=EVAL_OLLAMA_URL,
                temperature=0,
                request_timeout=120,
            )
        )
        embeddings = LangchainEmbeddingsWrapper(get_embeddings())

        _ragas_metrics = [
            Faithfulness(llm=llm),
            AnswerRelevancy(llm=llm, embeddings=embeddings),
            LLMContextPrecisionWithoutReference(llm=llm),
        ]
        logger.info(f"RAGAS metrics initialisés (judge: {EVAL_JUDGE_MODEL} @ {EVAL_OLLAMA_URL})")
    except Exception as e:
        logger.error(f"Impossible d'initialiser RAGAS metrics : {e}")
        _ragas_metrics = []

    return _ragas_metrics


async def _async_eval(question: str, answer: str, context_chunks: list[str]) -> dict:
    """Run each RAGAS metric individually (ragas >= 0.2 API)."""
    from ragas import SingleTurnSample

    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=context_chunks,
    )

    metrics = _get_ragas_metrics()
    metric_names = ["faithfulness", "answer_relevancy", "context_precision"]
    results: dict[str, float | None] = {}

    for metric, name in zip(metrics, metric_names):
        try:
            score = await metric.single_turn_ascore(sample)
            results[name] = round(float(score), 4) if score is not None else None
        except Exception as e:
            logger.warning(f"RAGAS metric '{name}' échouée : {e}")
            results[name] = None

    return results


def enqueue_eval(**kwargs) -> None:
    """Fire-and-forget : stocke les données brutes dans eval_queue pour évaluation différée."""
    if not EVAL_ENABLED:
        return
    try:
        from backend.core.eval_store import enqueue_raw
        enqueue_raw(
            collection=kwargs.get("collection", ""),
            pipeline_hash=kwargs.get("pipeline_hash", "unknown"),
            search_hash=kwargs.get("search_hash", "unknown"),
            question=kwargs.get("question", ""),
            answer=kwargs.get("answer", ""),
            context_chunks=kwargs.get("context_chunks", []),
            retrieval_ms=kwargs.get("retrieval_ms", 0),
        )
    except Exception as e:
        logger.warning(f"enqueue_eval : stockage échoué ({e})")


def _run_batch_sync(limit: int) -> dict:
    """Traite jusqu'à `limit` items en attente (appelé dans un thread de fond)."""
    global _batch_running
    from backend.core.eval_store import get_pending_batch, delete_queued, insert_eval

    _batch_running = True
    processed = 0
    errors = 0

    try:
        items = get_pending_batch(limit=limit)
        if not items:
            return {"processed": 0, "errors": 0}

        for item in items:
            t0 = time.monotonic()
            eval_error: str | None = None
            scores: dict[str, float | None] = {
                "faithfulness": None,
                "answer_relevancy": None,
                "context_precision": None,
            }

            # Limiter les inputs : top 3 chunks + answer[:500]
            eval_chunks = item["context_chunks"][:3]
            eval_answer = item["answer"][:500]

            try:
                scores = asyncio.run(_async_eval(item["question"], eval_answer, eval_chunks))
            except Exception as e:
                eval_error = str(e)
                logger.error(f"RAGAS eval item {item['id']} échouée : {e}")
                errors += 1

            eval_ms = (time.monotonic() - t0) * 1000

            try:
                insert_eval(
                    collection=item["collection"],
                    pipeline_hash=item.get("pipeline_hash", "unknown"),
                    search_hash=item.get("search_hash", "unknown"),
                    question=item["question"],
                    answer_preview=item["answer"],
                    context_preview="\n\n---\n\n".join(eval_chunks),
                    faithfulness=scores.get("faithfulness"),
                    answer_relevancy=scores.get("answer_relevancy"),
                    context_precision=scores.get("context_precision"),
                    retrieval_ms=item.get("retrieval_ms", 0),
                    eval_ms=eval_ms,
                    eval_model=EVAL_JUDGE_MODEL,
                    eval_error=eval_error,
                )
                delete_queued([item["id"]])
                processed += 1
                logger.info(
                    f"RAGAS eval stored [{item['id']}] : "
                    f"faith={scores.get('faithfulness')} "
                    f"rel={scores.get('answer_relevancy')} "
                    f"prec={scores.get('context_precision')} "
                    f"eval_ms={eval_ms:.0f}"
                )
            except Exception as e:
                logger.error(f"Impossible de sauvegarder eval item {item['id']} : {e}")
                errors += 1

    finally:
        _batch_running = False

    return {"processed": processed, "errors": errors}


def start_batch_eval(limit: int = 5) -> bool:
    """Lance le batch RAGAS en arrière-plan. Retourne False si déjà en cours."""
    global _batch_running
    if _batch_running:
        return False
    _executor.submit(_run_batch_sync, limit)
    return True


def is_batch_running() -> bool:
    return _batch_running
