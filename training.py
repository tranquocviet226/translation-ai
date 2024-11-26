from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, EncoderDecoderCache
from datasets import load_dataset

# Load dataset (hoặc thay bằng đường dẫn tới dữ liệu của bạn)
dataset = load_dataset("json", data_files={"train": "train.json", "validation": "valid.json"})

# Preprocess dataset
def preprocess_data(example):
    tokenizer = T5Tokenizer.from_pretrained("t5-small", legacy=False)
    input_text = example["input"]
    target_text = example["output"]
    
    model_inputs = tokenizer(input_text, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(target_text, max_length=512, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    
    return model_inputs

tokenized_dataset = dataset.map(preprocess_data, batched=True)

# Load model
model = T5ForConditionalGeneration.from_pretrained("t5-small")

# Training arguments
training_args = TrainingArguments(
    output_dir="./t5-finetuned",        # Thư mục lưu mô hình
    eval_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_dir="./logs",
    save_total_limit=2,
    push_to_hub=True,                 # Đặt True nếu muốn đẩy lên Hugging Face Hub
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=T5Tokenizer.from_pretrained("t5-small", legacy=False),  # Ensure correct tokenizer usage
)

# Fine-tune
trainer.train()
trainer.save_model("./result")