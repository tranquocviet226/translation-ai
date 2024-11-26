from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

app = Flask(__name__)

# Load pre-trained model and tokenizer
MODEL_NAME = "VietAI/envit5-translation"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

@app.route('/hello', methods=['GET'])
def hello():
    try:
        msg = {
            "id": 'hl-1',
            "text": 'Hello'
        }
        return jsonify(msg)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/translate/<string:text>', methods=['GET'])
def translate(text):
    try:
        # Parse input JSON
        # data = request.get_json()  # Get JSON data from request
        # if 'text' not in data:
        #     return jsonify({"error": "Missing 'text' in request body"}), 400
        
        # text = "This model does not have enough activity to be deployed to Inference API (serverless) yet. Increase its social visibility and check back later, or deploy to Inference Endpoints (dedicated) instead."

        prompt = f"Dịch từ 'available' trong câu: 'By default Corodomo will install support for all available languages'"

        # Check if CUDA is available and set device accordingly
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Tokenize and generate translation
        outputs = model.generate(tokenizer(prompt, return_tensors="pt", padding=True).input_ids.to(device), max_length=512)
        translated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return jsonify({"translated_text": translated_text})
    except Exception as e:
        print(f"Error: {str(e)}")  # Log the error for debugging
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
