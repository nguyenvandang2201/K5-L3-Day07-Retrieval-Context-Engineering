# Báo cáo Day 7 — Nguyễn Văn Đăng

**MSSV:** 2A202603371 · **Lớp:** L3A · **Chủ đề:** Agentic RAG trên Python 3.12 Tutorial

## 1. Dữ liệu, nguồn và benchmark

Bài làm dùng snapshot sáu chương HTML của Python 3.12 Tutorial từ [bài Day 6](https://github.com/nguyenvandang2201/K5-L3A-DAY06-NguyenVanDang-2A202603371-DataIngestion): 59 đơn vị nguồn và 274 chunk paragraph (tối đa 650 ký tự). `data/day6/manifest.json` ghi URL, thời điểm lấy, SHA-256, giấy phép Python Software Foundation License Version 2 và độ dài từng đơn vị; giấy phép được kèm tại `data/day6/PYTHON_LICENSE.txt`. `page` là số thứ tự đơn vị trong chương, **không phải số trang PDF**. Mỗi chunk giữ `source_id`, URL, đơn vị, metadata chương và ID ổn định.

Day 6 dùng `sentence-transformers/all-MiniLM-L6-v2`, 384 chiều, chuẩn hóa vector; bài này dùng lại đúng model và pgvector strategy `C` của Day 6. `scripts/index_day6.py` tái tạo cùng 274 chunk/vector trong PostgreSQL 16 + pgvector 0.8.7. Query embedding khớp model của index. Mọi tìm kiếm trên DB dùng tham số SQL; filter chỉ nhận `chapter` từ câu hỏi, không lấy từ gold.

`data/questions.json` có **20 câu dev** (18 có đáp án, 2 thiếu evidence) và **7 câu test** (5 có đáp án, 2 thiếu evidence). Năm câu test có đáp án lấy từ các đơn vị nguồn chưa xuất hiện trong benchmark Day 6; script `scripts/build_questions.py` kiểm tra evidence trong nguồn và trong chunk. Câu thiếu evidence được đặt gần chủ đề nhưng thông tin yêu cầu không có trong sáu chương; nhãn được kiểm tra ở mức corpus, không suy từ việc top-k không tìm được. Gold và expected answer chỉ được dùng trong `src/benchmark.py` hoặc audit, không truyền cho `src/agent.py` hay retrieval.

## 2. Ba phương pháp retrieval và context

1. **Dense:** embed query bằng MiniLM, exact cosine trên pgvector Day 6.
2. **BM25:** index lexical trên cùng 274 chunk, token hóa Unicode, xếp theo BM25.
3. **Hybrid:** lấy tối đa 30 candidate từ mỗi retriever và hợp nhất rank bằng reciprocal-rank fusion, hằng số 60. Không cộng trực tiếp score cosine với score BM25 vì khác thang đo.

Mỗi kết quả có ID, text, nguồn, đơn vị, metadata, score/rank. Context ghép các chunk theo thứ tự, bỏ ID trùng và giới hạn **ký tự**, gồm cả header nguồn. Top-3 luôn được xuất để so sánh ranking ngay cả khi cấu hình dùng top-6. `Hit@1`/`Evidence Hit@3` yêu cầu một chunk đúng nguồn/đơn vị và chứa nguyên văn evidence (sau chuẩn hóa khoảng trắng). Bảng dưới đo evidence trong **context đã đóng gói trước khi reader lọc thêm**; mẫu số là 18 dev hoặc 5 test có đáp án.

| Cấu hình | Phương pháp / top-k / budget | Giả thuyết | Split | Hit@1 | Hit@3 | Packed context evidence | Median retrieval |
|---|---|---|---|---:|---:|---:|---:|
| dense_base | Dense / 3 / 2400 ký tự | Baseline Day 6 | dev | 9/18 | 15/18 | 15/18 | 46,02 ms |
| bm25_base | BM25 / 3 / 2400 | Từ khóa tiếng Anh tăng xếp hạng | dev | 12/18 | 14/18 | 14/18 | 0,64 ms |
| hybrid_base | RRF / 3 / 2400 | Hợp nhất hai tín hiệu tăng hit | dev | 14/18 | 16/18 | 16/18 | 32,44 ms |
| bm25_wide | BM25 / 6 / 4000 | Thêm candidate giúp context | dev | 12/18 | 14/18 | 18/18 | 0,64 ms |
| **hybrid_wide** | **RRF / 6 / 4000** | **Giữ ranking hybrid và thêm context** | **dev** | **14/18** | **16/18** | **18/18** | **31,76 ms** |
| dense_base | Dense / 3 / 2400 | Baseline đã chốt | test | 4/5 | 4/5 | 4/5 | 43,35 ms |
| **hybrid_wide** | **RRF / 6 / 4000** | **Cấu hình đã chọn** | **test** | **4/5** | **4/5** | **5/5** | **41,10 ms** |

Các dòng so sánh cùng corpus, embedding/index, bộ câu hỏi và gold của split tương ứng. Không có API cost (0 USD), nhưng CPU/điện năng không được quy đổi thành tiền. Các median chỉ đo retrieval sau khi model/index đã tải, không gồm thời gian khởi tạo. Trên dev, `bm25_wide` và `hybrid_wide` là hai thử nghiệm tăng context so với baseline; `hybrid_base` là thử nghiệm hợp nhất xếp hạng. Việc tăng top-k không được coi là phương pháp retrieval mới. Quy tắc chọn được ghi vào `results/selection.json` **trước khi chạy test**: tối đa packed context evidence, tiếp đến Hit@3, Hit@1, cuối cùng thời gian median. Kết quả chọn `hybrid_wide`.

Reader chỉ nhận các chunk trong packed context vượt bộ lọc liên quan đến câu hỏi. Vì vậy **Context Evidence Rate của context thực sự gửi reader** có thể thấp hơn bảng trên: dense_base `15/18` dev và `4/5` test; bm25_base `10/18` dev; hybrid_wide `15/18` dev và `5/5` test. Chỉ số này được tính từ `candidate_ids` trong trace reader bởi `scripts/score_agent.py`. Hai cấu hình rộng khác không chạy agent nên chỉ báo packed context, không gán cho chúng Context Evidence Rate của reader. Sự chênh lệch 18/18 → 15/18 ở hybrid_wide là lỗi chọn context cần cải thiện ở thí nghiệm sau.

Audit context cho 5 câu test có đáp án: dense_base có **4 câu mức 2, 1 câu mức 1**; hybrid_wide có **5 câu mức 2**. Ở `t04`, dense context có `__name__` nhưng thiếu giá trị khi chạy module như script; hybrid đưa được đoạn `__main__` vào context. Các nhãn, lý do và trace nằm tại `results/test/<cấu hình>/reviews.json` và `traces.json`.

## 3. Agent, câu trả lời và citation

Agent dùng model local thật `Qwen/Qwen2.5-0.5B-Instruct` để tạo action `search` và query. Có tối đa hai lượt search; query trùng được dừng. Model `deepset/roberta-base-squad2` đọc các chunk có mặt trong context và trích span. Chỉ phát câu trả lời khi span có trong một chunk được giữ, kèm `[chunk-id]`; nếu không có span vượt ngưỡng thì từ chối. Điều kiện kiểm tra span/ID không khẳng định câu trả lời đúng về nghĩa: mỗi citation vẫn được audit với câu hỏi. Trace ghi action của model, query, filter, top-k, kết quả/score, context, proposal của reader, answer, citation, thời gian và lỗi. Provider/retriever lỗi có status `error`, không thay bằng kết quả giả.

Ba cấu hình dense_base, bm25_base, hybrid_wide đã chạy agent thật trên dev. Baseline và cấu hình chọn chạy cùng bảy câu test. Kết quả từ nhãn audit tại `reviews.json`:

| Cấu hình / split | Tool use | Số lượt search | Grounded 0/1/2; TB /2 | Citation hỗ trợ | Câu trả lời có citation | Abstention đúng | Từ chối nhầm | Median agent |
|---|---:|---|---|---:|---:|---:|---:|---:|
| dense_base / dev | 20/20 | 16 câu 1 lượt, 4 câu 2 lượt | 4/2/12; 1,44 | 14/17 | 17/17 | 2/2 | 1/18 | 1285 ms |
| bm25_base / dev | 20/20 | 13 câu 1 lượt, 7 câu 2 lượt | 5/2/11; 1,33 | 13/15 | 15/15 | 2/2 | 3/18 | 1268 ms |
| hybrid_wide / dev | 20/20 | 18 câu 1 lượt, 2 câu 2 lượt | 3/2/13; 1,56 | 15/18 | 18/18 | 2/2 | 0/18 | 1464 ms |
| dense_base / test | 7/7 | 7 câu 1 lượt | 2/1/2; 1,00 | 3/4 | 4/4 | 2/2 | 1/5 | 1520 ms |
| **hybrid_wide / test** | **7/7** | **7 câu 1 lượt** | **1/1/3; 1,40** | **4/5** | **5/5** | **2/2** | **0/5** | **1622 ms** |

Không có lỗi provider trong các lượt trên. Median agent gồm model lập query, retrieval và reader, chưa gồm tải model lúc khởi động. Citation hỗ trợ là citation trong context **và hỗ trợ câu trả lời cho câu hỏi**, vì vậy `t05` bị chấm không hợp lệ dù ID và span đều tồn tại. Các nhãn dev và test là một lượt đối chiếu theo nguồn, chưa có người chấm độc lập thứ hai.

### Ví dụ kiểm tra nguồn và lỗi

| ID | Loại | Tool/context và kết quả | Nhãn |
|---|---|---|---|
| `t04` | Có đáp án | Hybrid tìm `modules`, đơn vị 3; context chứa `with the __name__ set to "__main__"`; trả `__main__ [1bd3e0641bb9bd212411bb22]` | 2; citation hỗ trợ. Dense thiếu đoạn này và từ chối (0). |
| `t_abs01` | Thiếu evidence | Agent tìm nguồn nhưng không tìm được bằng chứng về thời hạn bảo hành Python; trả không đủ thông tin | Từ chối đúng. |
| `t05` | Có đáp án, lỗi | Câu hỏi hỏi lớp cha thường dùng của exception tự định nghĩa. Agent trích `BaseException` từ `errors`, đơn vị 4, trong khi `errors`, đơn vị 6 nói `Exception` trực tiếp/gián tiếp | 0; citation không hỗ trợ câu trả lời. |

Hai lỗi dev đã đọc và phân tích trước khi chốt agent test: `q03` nhầm backslash nối dòng với raw string; `q10` nhầm thao tác `pop()` với nguyên tắc LIFO của stack. `q12` cũng cho thấy cùng một chương có nhiều đoạn liên quan về dictionary, khiến reader chọn tính bất biến của khóa thay vì tính duy nhất. Việc mở rộng context tăng khả năng có evidence nhưng **không tự sửa lỗi đọc span**. Sau thay đổi từ câu trả lời sinh tự do sang reader trích span, agent không còn tạo citation cho đoạn nằm ngoài context; các lỗi ngữ nghĩa còn lại được giữ nguyên trong nhãn, không xóa hoặc chỉnh test để nâng điểm.

## 4. Tái lập và giới hạn

Lệnh chạy, dependencies và cách dựng lại index nằm trong [SOLUTION.md](../../SOLUTION.md). Toàn bộ kết quả query, context, tool call, answer và lỗi nằm trong `results/dev/` và `results/test/`; nhãn test được giữ từng câu. Khi chạy lại model, cần chấm lại, không tái dùng `reviews.json` mặc định. Test chỉ có 5 câu có đáp án nên sai một câu đã thay đổi tỷ lệ đáng kể. Corpus chỉ gồm sáu chương văn bản; `page` là đơn vị giả lập, không có OCR hay xử lý hình. Đánh giá exact quote bỏ sót bằng chứng diễn đạt tương đương và schema chưa chấm multi-hop. Model local giúp không tốn API nhưng không tính CPU/điện; context budget đo bằng ký tự, chưa ghi số token thực tiêu thụ. Bài này hoàn thành **phần cơ bản**; thử thách bonus tài liệu từ 100 trang và ngưỡng 90% test chưa thực hiện, nên không tự nhận 15 điểm bonus.
