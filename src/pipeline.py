from __future__ import annotations

"""Production RAG Pipeline — Ghép toàn bộ M1+M2+M3+M4+M5."""

import os, sys, time
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.m1_chunking import load_documents, chunk_hierarchical
from src.m2_search import HybridSearch
from src.m3_rerank import CrossEncoderReranker
from src.m4_eval import load_test_set, evaluate_ragas, failure_analysis, save_report
from src.m5_enrichment import enrich_chunks
from config import RERANK_TOP_K


def build_pipeline():
    """Build production RAG pipeline."""
    print("=" * 60)
    print("PRODUCTION RAG PIPELINE")
    print("=" * 60, flush=True)

    # Step 1: Load & Chunk (M1)
    t0 = time.time()
    print("\n[1/4] Chunking documents...", flush=True)
    docs = load_documents()
    all_chunks = []
    for doc in docs:
        parents, children = chunk_hierarchical(doc["text"], metadata=doc["metadata"])
        for child in children:
            all_chunks.append({"text": child.text, "metadata": {**child.metadata, "parent_id": child.parent_id}})
    print(f"  ✓ {len(all_chunks)} chunks from {len(docs)} documents ({time.time()-t0:.1f}s)", flush=True)

    # Step 2: Enrichment (M5)
    t0 = time.time()
    print(f"\n[2/4] Enriching {len(all_chunks)} chunks (M5, 1 API call/chunk)...", flush=True)
    enriched = enrich_chunks(all_chunks)
    if enriched:
        all_chunks = [{"text": e.enriched_text, "metadata": e.auto_metadata} for e in enriched]
        print(f"  ✓ Enriched {len(enriched)} chunks ({time.time()-t0:.1f}s)", flush=True)
    else:
        print("  ⚠️  M5 not implemented — using raw chunks", flush=True)

    # Step 3: Index (M2)
    t0 = time.time()
    print(f"\n[3/4] Indexing {len(all_chunks)} chunks (BM25 + Dense)...", flush=True)
    search = HybridSearch()
    search.index(all_chunks)
    print(f"  ✓ Indexed ({time.time()-t0:.1f}s)", flush=True)

    # Step 4: Reranker (M3)
    t0 = time.time()
    print("\n[4/4] Loading reranker...", flush=True)
    reranker = CrossEncoderReranker()
    print(f"  ✓ Reranker ready ({time.time()-t0:.1f}s)", flush=True)

    return search, reranker


def run_query(query: str, search: HybridSearch, reranker: CrossEncoderReranker) -> tuple[str, list[str]]:
    """Run single query through pipeline."""
    results = search.search(query)
    docs = [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in results]
    top_k_rerank = max(RERANK_TOP_K, 4)
    reranked = reranker.rerank(query, docs, top_k=top_k_rerank)
    contexts = [r.text for r in reranked] if reranked else [r.text for r in results[:top_k_rerank]]

    from config import OPENAI_API_KEY
    if OPENAI_API_KEY and contexts:
        try:
            from openai import OpenAI
            client = OpenAI()
            context_str = "\n\n".join(contexts)
            system_prompt = (
                "Bạn là trợ lý AI chuyên nghiệp giải đáp các thắc mắc về chính sách và quy định nội bộ của tổ chức dựa TRÊN CÁC ĐOẠN TRÍCH (Context) ĐƯỢC CUNG CẤP.\n\n"
                "CÁC NGUYÊN TẮC BẮT BUỘC:\n"
                "1. TÍNH TRUNG THỰC (Faithfulness):\n"
                "   - Trả lời CHỈ dựa trên dữ liệu có trong Context. Tuyệt đối không suy đoán hoặc bịa đặt thông tin không được đề cập.\n"
                "2. XỬ LÝ XUNG ĐỘT PHIÊN BẢN (Version Conflict):\n"
                "   - Nếu trong Context xuất hiện nhiều phiên bản chính sách khác nhau (ví dụ: v2023 và v2024, v1.0 và v2.0, cũ và mới), LUÔN LUÔN khẳng định câu trả lời theo chính sách HIỆN HÀNH / MỚI NHẤT (v2024, v2.0), đồng thời có thể ghi chú ngắn gọn chính sách cũ đã bị thay thế.\n"
                "3. ĐỘ PHÙ HỢP CÂU TRẢ LỜI (Answer Relevancy):\n"
                "   - Trả lời trực diện, đầy đủ câu với chủ thể rõ ràng, lặp lại các thực thể quan trọng trong câu hỏi để đảm bảo tính tường minh.\n"
                "4. CÂU HỎI PHỦ ĐỊNH / ĐIỀU KIỆN (Negation):\n"
                "   - Với câu hỏi có/không hoặc điều kiện loại trừ (nhân viên thử việc, nghỉ không lương, tự xử lý sự cố...), nếu quy định không cho phép, hãy nêu rõ 'KHÔNG' hoặc 'Không được phép' kèm lý do/điều kiện theo Context.\n"
                "5. CÂU HỎI ĐA BƯỚC & TÍNH TOÁN (Multi-hop & Numeric):\n"
                "   - Nếu câu hỏi có nhiều vế, hãy trả lời đầy đủ từng vế.\n"
                "   - Nếu cần tính toán (như ngày phép thâm niên), hãy trình bày rõ phép tính: [ngày phép cơ bản] + [ngày phép thâm niên] = [tổng số ngày].\n"
                "6. Nếu Context hoàn toàn không chứa thông tin để trả lời, trả lời: 'Không tìm thấy thông tin.'"
            )
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Context:\n{context_str}\n\nCâu hỏi: {query}"},
                ],
            )
            answer = resp.choices[0].message.content
        except Exception as e:
            print(f"  ⚠️  LLM generation failed: {e}", flush=True)
            answer = contexts[0]
    else:
        answer = contexts[0] if contexts else "Không tìm thấy thông tin."
    return answer, contexts


def evaluate_pipeline(search: HybridSearch, reranker: CrossEncoderReranker):
    """Run evaluation on test set."""
    test_set = load_test_set()
    print(f"\n[Eval] Running {len(test_set)} queries...", flush=True)
    questions, answers, all_contexts, ground_truths = [], [], [], []

    for i, item in enumerate(test_set):
        answer, contexts = run_query(item["question"], search, reranker)
        questions.append(item["question"])
        answers.append(answer)
        all_contexts.append(contexts)
        ground_truths.append(item["ground_truth"])
        print(f"  [{i+1}/{len(test_set)}] {item['question'][:50]}...", flush=True)

    t0 = time.time()
    print(f"\n[Eval] Running RAGAS (4 metrics × {len(test_set)} questions)...", flush=True)
    results = evaluate_ragas(questions, answers, all_contexts, ground_truths)
    print(f"  ✓ RAGAS done ({time.time()-t0:.1f}s)", flush=True)

    print("\n" + "=" * 60)
    print("PRODUCTION RAG SCORES")
    print("=" * 60)
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        s = results.get(m, 0)
        print(f"  {'✓' if s >= 0.75 else '✗'} {m}: {s:.4f}")

    failures = failure_analysis(results.get("per_question", []))
    save_report(results, failures)
    return results


if __name__ == "__main__":
    start = time.time()
    search, reranker = build_pipeline()
    evaluate_pipeline(search, reranker)
    print(f"\nTotal: {time.time() - start:.1f}s")
