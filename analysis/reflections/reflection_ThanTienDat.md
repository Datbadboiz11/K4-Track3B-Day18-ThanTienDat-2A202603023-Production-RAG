# Individual Reflection — Lab 18: Production RAG Pipeline

**Họ và tên học viên:** Thân Tiến Đạt  
**Mã số học viên:** 2A202603023  
**Khóa học:** K4 - Track 3B (AI Thực Chiến)  
**Ngày hoàn thành:** 04/10/2026  

---

## Phần 1: Mapping bài giảng (Lecture Mapping)

Bảng đối chiếu giữa các khái niệm lý thuyết cốt lõi trong bài giảng và việc hiện thực mã nguồn thực tế trong bài Lab 18:

| Khái niệm bài giảng (Lecture Concept) | Module | Hàm / Class cụ thể trong Code | Quan sát thực nghiệm & Phân tích chuyên sâu |
|:---|:---:|:---|:---|
| **Semantic Chunking** | **M1** | `chunk_semantic()` | Dùng `SentenceTransformer('all-MiniLM-L6-v2')` tính cosine similarity giữa các câu liên tiếp. Với ngưỡng `threshold=0.85`, các câu cùng một chủ đề được gom chặt chẽ, không bị cắt đứt mạch ý như phép cắt theo số ký tự cố định. Giúp bảo toàn ngữ nghĩa trọn vẹn cho từng đoạn trích. |
| **Hierarchical Chunking (Parent - Child)** | **M1** | `chunk_hierarchical()` | Tạo 2 cấp độ phân đoạn: Parent (2048 ký tự) chứa bức tranh toàn cảnh, Child (256 ký tự) phục vụ truy hồi chính xác. Thực nghiệm cho thấy khi tìm kiếm bằng Child vector và trả về Parent context, độ phủ ngữ cảnh tăng lên rõ rệt, hỗ trợ LLM nắm bắt đầy đủ điều kiện đi kèm. |
| **BM25 + Dense Fusion (RRF)** | **M2** | `reciprocal_rank_fusion()` | Kết hợp sức mạnh của tìm kiếm từ khóa chính xác (BM25 với tách từ `underthesea`) và tìm kiếm ngữ nghĩa sâu (Dense bge-m3 trên Qdrant). Thuật toán RRF với hệ số $k=60$ giúp trung hòa điểm số khác thang đo, đưa tài liệu được cả 2 phương pháp đánh giá cao lên đầu bảng. |
| **Cross-Encoder Reranking** | **M3** | `CrossEncoderReranker.rerank()` | Mô hình `BAAI/bge-reranker-v2-m3` xử lý đồng thời cặp `(query, document)`. Kết quả thực nghiệm đo được Context Precision đạt mức ấn tượng **0.9458** (+2.08% so với baseline), loại bỏ triệt để các văn bản nhiễu trước khi nạp vào cửa sổ ngữ cảnh của LLM. |
| **RAGAS 4 Metrics & Diagnostic Tree** | **M4** | `evaluate_ragas()` & `failure_analysis()` | Đánh giá độc lập 4 góc nhìn: Faithfulness (độ trung thực), Answer Relevancy (độ phù hợp câu trả lời), Context Precision (độ chuẩn xác ngữ cảnh) và Context Recall (độ đầy đủ ngữ cảnh). Cây chẩn đoán lỗi tự động phân loại đúng nguyên nhân (do Retrieval hay do Generation). |
| **Contextual Enrichment (Anthropic style)** | **M5** | `contextual_prepend()` / `_enrich_single_call()` | Bổ sung 1 câu tóm tắt vị trí tài liệu trước mỗi chunk văn bản. Chế độ Combined Single-Call gom cả Summarization, HyQA, Contextual Prepend và Metadata Extraction vào 1 API call/chunk giúp tiết kiệm 75% chi phí và thời gian gọi LLM. |

---

## Phần 2: Khó khăn & Cách giải quyết (Challenges & Debugging)

### 1. Lỗi xung đột môi trường Python toàn cục và Keras 3 / TensorFlow
* **Lỗi kỹ thuật gặp phải (Exact error message):**
  ```text
  ValueError: Your currently installed version of Keras is Keras 3, but this is not yet supported in Transformers. 
  Please install the backwards-compatible tf-keras package with `pip install tf-keras`.
  ```
* **Nguyên nhân gốc rễ:** 
  Khi thực hiện chạy lệnh kiểm thử `pytest tests/test_m3.py` trên Windows PowerShell, hệ điều hành đã tự động ưu tiên gọi trình thông dịch Python toàn cục (`C:\Program Files\Python312\python.exe`) và quét các thư viện trong `AppData\Roaming\Python\Python312\site-packages` (nơi có cài Keras 3 xung đột với HuggingFace Transformers), thay vì gọi Python trong môi trường ảo `.venv` của dự án.
* **Cách debug & xử lý:** 
  Chỉ định trực tiếp đường dẫn thực thi của môi trường ảo: `.venv\Scripts\python.exe -m pytest tests/test_m3.py -v`. Sau khi trỏ đúng môi trường cách ly, toàn bộ mô hình PyTorch và SentenceTransformers hoạt động ổn định và vượt qua 100% test cases.

### 2. Lỗi phân tách từ ghép tiếng Việt cho BM25
* **Vấn đề gặp phải:** Thư viện `underthesea.word_tokenize(format="text")` tự động nối các từ ghép tiếng Việt bằng dấu gạch dưới `_` (ví dụ: `"nghỉ_phép_năm"`). Trong khi đó, bộ tokenizer của thư viện `rank_bm25` chỉ phân tách từ theo khoảng trắng (`split()`). Do đó, câu query của người dùng gõ `"nghỉ phép năm"` sẽ bị tách thành 3 token riêng biệt (`["nghỉ", "phép", "năm"]`), dẫn đến không khớp với token `"nghỉ_phép_năm"` trong chỉ mục, làm điểm số BM25 bị tụt về 0.
* **Cách khắc phục:** Trong hàm `segment_vietnamese()`, thực hiện thay thế toàn bộ ký tự `_` thành dấu cách: `segmented.replace("_", " ")`. Nhờ vậy, cả dữ liệu chỉ mục và câu truy vấn đều đồng nhất không gian từ khóa.

### 3. Thử thách về xung đột phiên bản tài liệu (Version Conflict)
* **Hiện tượng:** Trong tập dữ liệu có các văn bản quy định cũ và mới song song (`nghi_phep_nam_v2023.md` 12 ngày vs `nghi_phep_nam_v2024.md` 15 ngày; `mat_khau_v1.md` 90 ngày vs `mat_khau_v2.md` 120 ngày). Tầng Retrieval thu hồi cả 2 phiên bản và LLM bị trả lời nhầm sang chính sách cũ.
* **Cách khắc phục cho tương lai:** Cần kết hợp Metadata Filtering theo trạng thái tài liệu (`status: active` / `superseded`) ngay từ bước truy hồi, đồng thời tinh chỉnh prompt ép buộc LLM ưu tiên văn bản hiện hành.

---

## Phần 3: Action Plan cho Project cá nhân (Application Plan)

### Dự án: Hệ thống Trợ lý AI Hỏi đáp Tài liệu Kỹ thuật & Chính sách Doanh nghiệp (Enterprise RAG Assistant)

#### 1. Hiện trạng & Điểm nghẽn (Current Bottlenecks)
* **Kiến trúc hiện tại:** Đang dùng Naive RAG cơ bản với phép cắt đoạn RecursiveCharacterTextSplitter cố định và tìm kiếm Dense Cosine thuần túy trên vector database.
* **Vấn đề đang gặp:**
  1. *Hallucination cao:* Khi gặp câu hỏi về điều khoản ngoại lệ hoặc số liệu kỹ thuật, mô hình thường tự suy diễn do chunk bị cắt cụt giữa chừng.
  2. *Low Keyword Precision:* Không tìm kiếm chính xác được các mã lỗi (error codes), ký hiệu phiên bản phần mềm hoặc các thuật ngữ chuyên ngành viết tắt.
  3. *Context Flooding:* Top-10 vector candidates chứa nhiều đoạn trích trùng lặp hoặc không liên quan trực tiếp, làm loãng câu trả lời của LLM.

#### 2. Kế hoạch áp dụng các kỹ thuật từ Lab 18
1. **Chiến lược Chunking:**
   * Áp dụng **Hierarchical Chunking (Parent-Child)**: Cắt các tài liệu hướng dẫn kỹ thuật dài thành Child chunks (150 - 200 từ) để vector search có độ nhạy cao, nhưng khi nạp vào LLM sẽ lấy toàn bộ Parent chunk (khoảng 800 - 1000 từ) để đảm bảo không mất ngữ cảnh điều kiện.
   * Áp dụng **Structure-Aware Chunking** cho các tài liệu API docs và bảng biểu Markdown để không làm vỡ cấu trúc bảng.
2. **Hạ tầng Tìm kiếm (Search & Retrieval):**
   * Triển khai **Hybrid Search**: Kết hợp BM25 (xử lý mã lỗi, tên biến, số hiệu quy trình) với Dense Qdrant Embeddings (`BAAI/bge-m3`).
   * Sử dụng thuật toán **RRF** ($k=60$) để tự động cân bằng thứ hạng mà không cần phải thủ công tinh chỉnh trọng số $\alpha$.
3. **Tái xếp hạng (Reranking):**
   * Bắt buộc tích hợp tầng **Cross-Encoder Reranker** (`BAAI/bge-reranker-v2-m3`). Lọc từ 25 candidate ban đầu xuống còn Top 3 - 5 context chất lượng nhất trước khi gửi vào LLM generation.
4. **Làm giàu dữ liệu (Data Enrichment):**
   * Ứng dụng kỹ thuật **Contextual Prepend (Anthropic style)** trước khi sinh vector: Tự động bổ sung tiêu đề chương mục và vai trò của đoạn trích vào đầu mỗi chunk để khắc phục tình trạng chunk đứng độc lập bị mất ngữ cảnh nguồn.
   * Chạy chế độ **Combined Single-Call** lúc offline ingestion để tiết kiệm chi phí API.
5. **Khung Đánh giá & Giám sát liên tục:**
   * Xây dựng bộ test benchmark 50 câu hỏi đặc thù và tích hợp **RAGAS** vào quy trình CI/CD. Đặt ngưỡng chặn (Quality Gate): Faithfulness $\ge 0.85$, Context Precision $\ge 0.90$.

#### 3. Lộ trình triển khai cụ thể (Timeline)
* **Tuần 1:** Tái cấu trúc pipeline Ingestion: Cài đặt Hierarchical Chunking, Structure-Aware Chunking và Contextual Prepend cho toàn bộ kho tài liệu nội bộ.
* **Tuần 2:** Dựng cụm Qdrant kết hợp BM25 tiếng Việt có xử lý token chuẩn hóa; kiểm thử thuật toán RRF.
* **Tuần 3:** Tích hợp Cross-Encoder Reranker; tối ưu hóa Prompt template (hướng dẫn Chain-of-Thought và xử lý xung đột phiên bản quy định).
* **Tuần 4:** Thiết lập luồng kiểm thử tự động RAGAS benchmark trên môi trường Staging và đóng gói thành Production Docker container.
