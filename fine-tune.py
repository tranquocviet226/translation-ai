from transformers import MarianMTModel, MarianTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset, DatasetDict

# Định nghĩa model và tokenizer
model_name = "VietAI/envit5-translation"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Load song ngữ từ file
def load_bilingual_dataset(src_file, tgt_file):
    with open(src_file, "r", encoding="utf-8") as f_src, open(tgt_file, "r", encoding="utf-8") as f_tgt:
        src_lines = f_src.readlines()
        tgt_lines = f_tgt.readlines()

    assert len(src_lines) == len(tgt_lines), "Số dòng không khớp giữa src và tgt!"
    return Dataset.from_dict({"src_texts": src_lines, "tgt_texts": tgt_lines})

# Load train và validation
raw_datasets = DatasetDict({
    "train": load_bilingual_dataset("train.src", "train.tgt"),
    "validation": load_bilingual_dataset("valid.src", "valid.tgt"),
})

# Tiền xử lý
def preprocess_function(examples):
    inputs = tokenizer(examples["src_texts"], max_length=128, truncation=True, padding="max_length")
    targets = tokenizer(examples["tgt_texts"], max_length=128, truncation=True, padding="max_length")
    inputs["labels"] = targets["input_ids"]
    return inputs

tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)

# Tham số training
training_args = Seq2SeqTrainingArguments(
    output_dir="./fine_tuned_opus_mt_en_vi",
    eval_strategy="epoch",  # Đổi từ evaluation_strategy thành eval_strategy
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    save_total_limit=2,
    num_train_epochs=3,
    predict_with_generate=True,
    logging_dir="./logs",
    logging_strategy="steps",
    logging_steps=100,
    save_strategy="epoch",
    warmup_steps=500,
    gradient_accumulation_steps=2,
    report_to="none",
    fp16=False  # Nếu sử dụng GPU hỗ trợ FP16
)

# Định nghĩa trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,  # Thay vì truyền tokenizer
)

# Train model
trainer.train()

# Lưu model
model.save_pretrained("./fine_tuned_opus_mt_en_vi")
tokenizer.save_pretrained("./fine_tuned_opus_mt_en_vi")
