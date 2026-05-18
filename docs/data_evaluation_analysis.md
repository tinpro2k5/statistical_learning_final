những vấn đề quan trọng về sự chênh lệch dữ liệu Test giữa 2 model (`Baseline` và `Hardx2`) và cách đánh giá đúng

## 1. Nguyên nhân sự chênh lệch số lượng mẫu ở tập Test
Trong file `training_summary.json` của hai model có sự chênh lệch lớn về `num_samples` ở tập Test:
- **Baseline (Combined):** 4165 samples
- **Hardx2:** 7497 samples

**Lý do chi tiết:**
- Code `preprocess_data.py` được thiết kế cực kỳ chặt chẽ: Việc chia tập test được tách ra **trước tiên** bằng ID câu hỏi, và cả 2 lần sinh data đều dùng chung `--seed 42`. Do đó, **các câu đúng (positive samples) ở 2 bên là hoàn toàn giống nhau 100%** (833 positives).
- Tuy nhiên, độ dài file test bị độn lên khác nhau do số lượng đáp án nhiễu (random negatives):
  - **Baseline:** Mặc định `--negatives_per_positive 4`. Vì tập Test ép `n_hard = 0`, nên mỗi câu đúng có 4 câu nhiễu. Số lượng pair: `833 * (1 + 4) = 4165`.
  - **Hardx2:** Chạy lệnh với `--negatives_per_positive 8`. Tương tự, mỗi câu đúng có 8 câu nhiễu. Số lượng pair: `833 * (1 + 8) = 7497`.

=> **Kết luận:** Tập Test của Hardx2 dài hơn đơn giản là vì nó phải gánh số lượng đáp án nhiễu gấp đôi.

## 2. Tại sao Tập Test lại không có Hard Negative?
Dù lúc train model có học với Hard Negative (Baseline là 2, Hardx2 là 4), nhưng tập Test của cả 2 hoàn toàn không có Hard Negative.
- Nguyên nhân là do trong code `preprocess_data.py`, `n_hard = 0` khi sinh tập test.
- Đây là một quyết định **hoàn toàn đúng đắn**. Tập test cần phải mô phỏng lại một tập dữ liệu "khách quan" tự nhiên. Nếu chèn ép hard negative vào test theo một tỷ lệ nhân tạo, điểm số sẽ bị bóp méo và không phản ánh đúng thực tế.

## 3. Vì sao điểm của Hardx2 lại thấp hơn Baseline? (0.9201 vs 0.9576)
- Điểm Hardx2 bị thấp đi **KHÔNG PHẢI** vì model học kém hơn. 
- Nguyên nhân cốt lõi là do model Hardx2 đang phải thi một cái đề bài khó hơn rất nhiều: Nó phải tìm ra 1 câu đúng giữa **8 câu nhiễu**, trong khi Baseline chỉ phải tìm ra 1 câu đúng giữa **4 câu nhiễu**.
- Càng nhiều đáp án nhiễu (distractors), xác suất xếp hạng đúng càng khó, dẫn đến NDCG và MRR bị tụt.

## 4. Reranking hoàn toàn KHÔNG SỢ mất cân bằng nhãn (Imbalance)
- Trong bài toán **Classification**, nếu tỉ lệ nhãn 1:8, model có thể "lười biếng" đoán tất cả là 0 và đạt Accuracy cao. Khi đó độ đo bị hỏng.
- Trong bài toán **Document Reranking**, model làm nhiệm vụ **Chấm điểm (Scoring)** và **Xếp hạng (Sorting)**. Độ đo (NDCG, MRR) chấm điểm dựa trên thứ hạng của văn bản đúng. Việc huấn luyện model với số lượng negative lớn (1:8) thực ra là một **điểm cộng rất lớn**, giúp ép model đối mặt với nhiều văn bản sai khác nhau để rèn luyện bộ đặc trưng ngữ nghĩa tốt hơn.


Vì các câu đúng ở 2 tập test là y hệt nhau, để so sánh công bằng tuyệt đối (Fair Comparison), chúng ta chỉ cần bắt 2 model chấm điểm chung một cái đề thi có cùng số lượng đáp án nhiễu.

**Cách thực hiện:**
Dùng script `evaluate.py` để chạy lại điểm cho cả 2 model trên cùng file `data/combined/test.jsonl` (file 4165 mẫu của Baseline):
