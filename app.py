from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load model and tokenizer
# model_name = "google-t5/t5-small"
model_name="tranquocviet226/t5-finetuned"
# model_name = T5ForConditionalGeneration.from_pretrained("")
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)

# Input text for translation
text = "Tôi thích xem phim"

# Tokenize input text
inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)

# Generate translation
outputs = model.generate(**inputs, max_length=512, num_beams=4, early_stopping=True)

# Decode the output
translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(translated_text)
