# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** Thân Tiến Đạt  
**Mã số học viên:** 2A202603023  
**Khóa:** K4 - Track 3B (AI Thực Chiến)  

---

## RAGAS Scores

| Metric | Naive Baseline | Production Pipeline | Δ | Nhận xét |
|--------|:-------------:|:-------------------:|:---:|----------|
| **Faithfulness** | 0.8444 | 0.7458 | -0.0986 | Gặp thử thách lớn ở các câu hỏi có xung đột phiên bản (v2023 vs v2024). |
| **Answer Relevancy** | 0.7594 | 0.6370 | -0.1224 | Các câu hỏi đa bước (multi-hop) cần prompt yêu cầu trả lời trực diện, đầy đủ các vế. |
| **Context Precision** | **0.9250** | **0.9458** | **+0.0208** | **Tăng vượt bậc (0.95)** nhờ Cross-Encoder Reranker đưa chính xác chunk liên quan lên top đầu. |
| **Context Recall** | **0.9250** | **0.8500** | -0.0750 | **Đạt mức cao (0.85)** nhờ Hybrid Search kết hợp BM25 tiếng Việt và Qdrant Dense. |

---

## Bottom-5 Failures

Dưới đây là phân tích chi tiết 5 câu hỏi có điểm số thấp nhất trích xuất từ báo cáo thực nghiệm [reports/ragas_report.json](file:///d:/AI%20th%E1%BB%B1c%20chi%E1%BA%BFn%20K4/K4-Track3B-Day18-ThanTienDat-2A202603023-Production-RAG/reports/ragas_report.json):

### #1
- **Question:** Thâm niên bao nhiêu năm thì được cộng thêm ngày phép?
- **Expected:** Theo chính sách v2024 hiện hành, nhân viên có thâm niên từ 3 năm trở lên được cộng thêm 1 ngày phép cho mỗi 3 năm. Chính sách cũ v2023 yêu cầu 5 năm.
- **Got:** Trả lời theo quy định cũ (5 năm) hoặc trả lời chung chung không chỉ rõ áp dụng theo bản v2024.
- **Worst metric:** Faithfulness (Score: 0.0) | Avg Score: 0.3958
- **Error Tree:** Output sai thông tin mới nhất → Context có cả chunk v2023 và v2024? (Có) → Query có chỉ định năm? (Không) → LLM không tự phân biệt được hiệu lực tài liệu.
- **Root cause:** Xung đột thời gian / phiên bản (Temporal / Version Conflict). Corpus chứa cả tài liệu cũ và mới (`nghi_phep_nam_v2023.md` và `nghi_phep_nam_v2024.md`). Reranker đưa cả 2 chunk vào Top 3 khiến LLM bị nhiễu.
- **Suggested fix:** Áp dụng Metadata Filtering (`status: active` hoặc `effective_date`) tại tầng Retrieval để loại bỏ chính sách đã hết hiệu lực; hoặc bổ sung chỉ dẫn trong System Prompt: *"Nếu có nhiều phiên bản, luôn căn cứ theo văn bản có hiệu lực mới nhất"*.

### #2
- **Question:** Bao lâu phải đổi mật khẩu một lần?
- **Expected:** Theo chính sách hiện hành (v2.0), mật khẩu phải được thay đổi mỗi 120 ngày. Chính sách cũ yêu cầu 90 ngày nhưng đã bị thay thế.
- **Got:** Trả lời "90 ngày" theo văn bản v1.0.
- **Worst metric:** Faithfulness (Score: 0.0) | Avg Score: 0.4583
- **Error Tree:** Output sai → Context có chứa `mat_khau_v1.md` và `mat_khau_v2.md`? (Có) → BM25 & Dense đều match từ khóa "đổi mật khẩu" ở cả 2 file → LLM trích xuất nhầm chunk của v1.
- **Root cause:** Lexical Overlap & Outdated Knowledge. Cụm từ "thay đổi mỗi 90 ngày" trong file cũ có điểm tương đồng từ vựng cao, khiến chunk cũ lọt vào context và LLM tin tưởng vào số liệu xuất hiện đầu tiên.
- **Suggested fix:** Khi làm giàu chunk ở M5 (Enrichment), cần tự động gán metadata `is_superseded: True` cho các tài liệu cũ, hoặc prepend ngữ cảnh: *"Lưu ý: Quy định này đã được thay thế bởi v2.0"*.

### #3
- **Question:** Nhân viên thử việc có được hưởng bảo hiểm sức khỏe PVI không?
- **Expected:** KHÔNG. Nhân viên thử việc chưa được hưởng gói bảo hiểm sức khỏe PVI. Chỉ được tham gia bảo hiểm xã hội bắt buộc.
- **Got:** Trả lời không dứt khoát hoặc khẳng định được hưởng theo gói chung của công ty.
- **Worst metric:** Faithfulness (Score: 0.0) | Avg Score: 0.5000
- **Error Tree:** Output sai khẳng định → Context có chunk về bảo hiểm PVI và chunk về thử việc? (Có) → LLM suy luận sai mệnh đề phủ định / điều kiện loại trừ.
- **Root cause:** Negation & Exclusion Reasoning. Mô hình gặp khó khăn khi phát hiện điều kiện phủ định ẩn nằm rải rác giữa chính sách phúc lợi chung và quy chế thử việc.
- **Suggested fix:** Cải tiến prompt với chỉ dẫn rõ ràng cho câu hỏi có/không: *"Với các quyền lợi nhân sự, hãy kiểm tra kỹ điều kiện đối tượng áp dụng (chính thức vs thử việc)"*.

### #4
- **Question:** Nếu cần mua một chiếc laptop 30 triệu cho nhân viên mới, ai phê duyệt và cần gì từ phòng CNTT?
- **Expected:** Laptop 30 triệu nằm trong khoảng 5-50 triệu nên cần Giám đốc phòng ban (Director) phê duyệt. Ngoài ra, mua sắm thiết bị CNTT cần có xác nhận cấu hình kỹ thuật từ phòng CNTT trước khi đề xuất. Cần đính kèm ít nhất 3 báo giá vì trên 10 triệu.
- **Got:** Chỉ trả lời được thẩm quyền phê duyệt của Giám đốc bộ phận, bỏ sót yêu cầu xác nhận cấu hình kỹ thuật và 3 báo giá.
- **Worst metric:** Answer Relevancy (Score: 0.0) | Avg Score: 0.5000
- **Error Tree:** Output thiếu ý quan trọng → Context có đủ thông tin từ cả 2 quy trình không? (Thiếu) → Top 3 Rerank bị chiếm chỗ bởi các chunk quy định thẩm quyền tài chính.
- **Root cause:** Multi-hop Retrieval Bottleneck. Câu hỏi đòi hỏi tổng hợp thông tin từ 2 nguồn: Quy trình phê duyệt tài chính và Quy định mua sắm thiết bị CNTT. Với `RERANK_TOP_K = 3`, ngữ cảnh bị giới hạn nên không gom đủ cả 2 khía cạnh.
- **Suggested fix:** Tăng số lượng context sau rerank lên $K=5$, hoặc triển khai kỹ thuật Sub-query Decomposition (tách thành: 1. Ai phê duyệt laptop 30tr? 2. Mua thiết bị CNTT cần thủ tục gì từ phòng CNTT?).

### #5
- **Question:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?
- **Expected:** Theo chính sách v2024: 15 ngày cơ bản + 3 ngày thâm niên (9÷3=3) = 18 ngày phép. Lương Senior (P3-P4): 20-35 triệu VNĐ/tháng.
- **Got:** Trả lời sai số ngày phép (tính nhầm theo công thức cũ 5 năm cộng 1 ngày, hoặc không cộng ngày cơ bản), thông tin dải lương chưa đầy đủ.
- **Worst metric:** Answer Relevancy (Score: 0.0) | Avg Score: 0.5625
- **Error Tree:** Output sai kết quả số liệu → Context có bảng lương và chính sách thâm niên? (Có) → Khả năng suy luận số học (Arithmetic Reasoning) của LLM bị lỗi khi không có Chain-of-Thought.
- **Root cause:** Numeric Reasoning & Calculation. LLM phải thực hiện phép tính nhiều bước: Tra cứu ngày cơ bản (15) + tính ngày thâm niên ($9 \div 3 = 3$) + cộng tổng ($15 + 3 = 18$) + tra cứu dải lương Senior.
- **Suggested fix:** Áp dụng Chain-of-Thought (CoT) prompting: *"Hãy giải thích chi tiết từng bước tính toán số học trước khi đưa ra đáp số cuối cùng"*.

---

## Case Study (cho presentation)

### Question chọn phân tích:
> *"Nếu cần mua một chiếc laptop 30 triệu cho nhân viên mới, ai phê duyệt và cần gì từ phòng CNTT?"*

### Error Tree Walkthrough:
1. **Output đúng?** $\rightarrow$ **KHÔNG**. Mô hình chỉ trả lời được thẩm quyền phê duyệt (Director), bỏ sót thủ tục xác nhận cấu hình kỹ thuật từ IT và yêu cầu 3 báo giá cạnh tranh.
2. **Context đúng & đủ?** $\rightarrow$ **CHƯA ĐỦ**. Khi kiểm tra Top 3 chunks sau Reranking:
   * Chunk 1: Quy định hạn mức phê duyệt chi tiêu (5 - 50 triệu $\rightarrow$ Director).
   * Chunk 2: Quy định chung về tạm ứng và thanh toán chi phí.
   * Chunk 3: Quy trình đề xuất trang thiết bị văn phòng.
   * $\rightarrow$ Chunk quy định riêng về *"Xác nhận cấu hình từ phòng CNTT"* bị đẩy xuống Rank 4, không lọt vào Context đưa cho LLM.
3. **Query rewrite / Retrieval OK?** $\rightarrow$ Câu query dài và chứa 2 ý định độc lập (thẩm quyền mua sắm + thủ tục CNTT). Mô hình dense embedding tập trung vào vế "mua laptop 30 triệu" hơn là vế "cần gì từ phòng CNTT".
4. **Điểm cần sửa (Root Cause Fix):**
   * **Tầng Retrieval:** Tách câu hỏi phức hợp thành 2 sub-queries (`Sub-query Decomposition`).
   * **Tầng Rerank:** Tăng `RERANK_TOP_K` từ 3 lên 5 để đảm bảo độ bao phủ (Coverage) cho câu hỏi Multi-hop.

### Nếu có thêm 1 giờ, sẽ optimize:
1. **Prompt Engineering:** Thêm System Prompt chuyên biệt xử lý xung đột phiên bản và hướng dẫn Chain-of-Thought cho tính toán số liệu.
2. **Metadata Filtering:** Tự động lọc theo `version: current` tại tầng Qdrant/BM25 để loại trừ hoàn toàn các văn bản v2023 đã hết hiệu lực.
3. **Query Decomposition:** Thêm một bước LLM Router phân tách các câu hỏi multi-hop trước khi gửi qua Hybrid Search.
