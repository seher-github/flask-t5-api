
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import T5Tokenizer, T5ForConditionalGeneration

app = Flask(__name__)
CORS(app)

model_path = "path/to/your/model"
model = T5ForConditionalGeneration.from_pretrained(model_path, use_safetensors=True)
tokenizer = T5Tokenizer.from_pretrained(model_path)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    input_text = data.get("input")
    if not input_text:
        return jsonify({"error": "No input provided"}), 400

    input_ids = tokenizer.encode(input_text, return_tensors="pt", truncation=True)
    output_ids = model.generate(input_ids, max_length=512)
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return jsonify({"output": output_text})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
