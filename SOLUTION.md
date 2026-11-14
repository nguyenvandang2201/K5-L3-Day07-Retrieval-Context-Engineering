# Bài lab Day 7 — Nguyễn Văn Đăng (2A202603371)

Bài làm nối lại snapshot Python 3.12 Tutorial và cấu hình chunk/embedding/vector store của [Day 6](https://github.com/nguyenvandang2201/K5-L3A-DAY06-NguyenVanDang-2A202603371-DataIngestion). Các file trong `data/day6/` gồm corpus, manifest, benchmark gốc để truy xuất nguồn và giấy phép PSF. `page` trong dữ liệu này là **số thứ tự đơn vị văn bản**, không phải trang PDF. Không đưa gold, đáp án chuẩn hay nhãn đánh giá vào agent.

## Thành phần

| Thành phần | File |
|---|---|
| Index Day 6: đoạn theo paragraph ≤650 ký tự, MiniLM 384 chiều, PostgreSQL/pgvector | `src/chunking.py`, `src/embeddings.py`, `src/store.py`, `scripts/index_day6.py` |
| Dense, BM25 và hybrid RRF (k=60) | `src/retrieval.py` |
| Chấm Hit@1, Evidence Hit@3, Context Evidence Rate | `src/benchmark.py` |
| Agent: Qwen2.5-0.5B-Instruct lập truy vấn; model SQuAD2 trích span từ chunk, có citation hoặc abstain | `src/agent.py`, `src/qa.py` |
| So sánh dev, chốt cấu hình, chạy test và tổng hợp nhãn | `scripts/run_dev.py`, `scripts/run_test.py`, `scripts/score_agent.py` |

Hai model chạy local trên CPU; lần đầu cần mạng để tải trọng số. Không dùng API trả phí. Qwen tạo tool action và query, agent giới hạn tối đa hai lượt tìm; câu trả lời của reader phải là span của chunk trong context. Kiểm tra span và ID chỉ là điều kiện kỹ thuật; nhãn `reviews.json` vẫn đối chiếu nội dung với nguồn.

## Tái lập

Yêu cầu Python 3.12, Docker/Compose và khoảng trống cho model local. Chạy từ thư mục gốc của repo:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
docker compose -p day7_lab up -d --wait
python -m scripts.index_day6
python -m scripts.build_questions
python -m scripts.run_dev
python -m src.agent --split dev --configuration dense_base --output results/dev/dense_base/agent_traces.json
python -m src.agent --split dev --configuration bm25_base --output results/dev/bm25_base/agent_traces.json
python -m src.agent --split dev --configuration hybrid_wide --output results/dev/hybrid_wide/agent_traces.json
python -m scripts.run_test
python -m src.agent --split test --configuration dense_base --output results/test/dense_base/agent_traces.json
python -m src.agent --split test --configuration hybrid_wide --output results/test/hybrid_wide/agent_traces.json
python -m unittest discover -s tests -v
```

Nếu đã chạy Day 6 và DB tại `127.0.0.1:55433` còn chứa strategy `C`, có thể dùng lại DB và chạy `scripts.index_day6` để xác nhận tái lập. Nếu cổng này đang được Day 6 sử dụng, không khởi động thêm Compose Day 7 trên cùng cổng. Có thể đặt `DATABASE_URL` cho một DB pgvector khác; khi đó thay port trong Compose tương ứng. Password trong Compose chỉ dành cho DB local demo.

`scripts.run_dev` ghi năm cấu hình và chọn theo quy tắc cố định trước test: evidence trong packed context, rồi Hit@3, Hit@1, cuối cùng thời gian median. `scripts/score_agent.py` báo riêng evidence trong các chunk thực sự gửi reader. Script test đọc `results/selection.json`; không chọn lại theo test. Mọi lần chạy ghi đè output cùng tên. `reviews.json` là nhãn audit của lần chạy được nộp; **không tái sử dụng nhãn đó nếu chạy lại model hoặc thay cấu hình**. Chấm lại từng câu rồi dùng:

```powershell
python -m scripts.score_agent --split test --traces results/test/hybrid_wide/agent_traces.json --reviews results/test/hybrid_wide/reviews.json --output results/test/hybrid_wide/agent_summary.json
```

## Kết quả và giới hạn

Kết quả đo, phân tích lỗi và nhãn từng câu nằm trong [báo cáo cá nhân](report/individual/2A202603371_NguyenVanDang.md) và `results/`. Bộ test nhỏ nên không suy rộng sang mọi tài liệu. Agent có thể trích một span đúng văn bản nhưng trả lời sai câu hỏi; ví dụ `t05` được chấm 0 và citation không hợp lệ về mặt hỗ trợ. `results/` giữ trace, nhãn, câu trả lời và thời gian để kiểm tra, không chứa API key hay `.env`.
