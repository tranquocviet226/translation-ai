from sklearn.model_selection import train_test_split

# Hàm làm sạch dữ liệu
def clean_lines(lines):
    return [line.strip() for line in lines if len(line.strip()) > 0 and len(line.strip().split()) < 100]

# Đọc file song ngữ
with open("OpenSubtitles.en-vi.en", "r", encoding="utf-8") as f_en:
    en_lines = f_en.readlines()

with open("OpenSubtitles.en-vi.vi", "r", encoding="utf-8") as f_vi:
    vi_lines = f_vi.readlines()

# Kiểm tra độ dài 2 file phải khớp
assert len(en_lines) == len(vi_lines), "File EN và VI không khớp số lượng dòng!"

print(f"Số lượng câu ban đầu: {len(en_lines)}")

# Làm sạch dữ liệu
en_lines_clean = clean_lines(en_lines)
vi_lines_clean = clean_lines(vi_lines)

# Loại bỏ các dòng không đồng bộ sau khi làm sạch
en_lines_clean, vi_lines_clean = zip(*[
    (en, vi) for en, vi in zip(en_lines_clean, vi_lines_clean) if len(en) > 0 and len(vi) > 0
])

print(f"Số lượng câu sau làm sạch: {len(en_lines_clean)}")

# Tách dữ liệu thành train và validation
train_en, valid_en, train_vi, valid_vi = train_test_split(
    en_lines_clean, vi_lines_clean, test_size=0.1, random_state=42
)

print(f"Số câu trong tập train: {len(train_en)}")
print(f"Số câu trong tập validation: {len(valid_en)}")

# Ghi tập train vào file
with open("train.src", "w", encoding="utf-8") as f_src:
    f_src.writelines(f"{line}\n" for line in train_en)

with open("train.tgt", "w", encoding="utf-8") as f_tgt:
    f_tgt.writelines(f"{line}\n" for line in train_vi)

with open("train.src", "r", encoding="utf-8") as f:
    train_src_lines = f.readlines()
with open("train.tgt", "r", encoding="utf-8") as f:
    train_tgt_lines = f.readlines()

assert len(train_src_lines) == len(train_tgt_lines), "Số dòng giữa train.src và train.tgt không khớp!"

# Ghi tập validation vào file
with open("valid.src", "w", encoding="utf-8") as f_src:
    f_src.writelines(f"{line}\n" for line in valid_en)

with open("valid.tgt", "w", encoding="utf-8") as f_tgt:
    f_tgt.writelines(f"{line}\n" for line in valid_vi)

print("Hoàn tất xử lý và tách dữ liệu!")

assert all(len(line.strip()) > 0 for line in train_src_lines), "Có dòng trống trong train.src!"
assert all(len(line.strip()) > 0 for line in train_tgt_lines), "Có dòng trống trong train.tgt!"

