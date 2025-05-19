# app.py
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
from cnn import classify_image
from chatbot import run_chatbot

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/classify")
def classify_page():
    return render_template("classify.html")

@app.route("/api/classify", methods=["POST"])
def api_classify():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    file = request.files["image"]
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        waste_type, recyclability = classify_image(filepath)
        
        return jsonify({
            "waste_type": waste_type,
            "recyclability": recyclability,
            "image_url": filepath
        })
    
    return jsonify({"error": "Invalid file type"}), 400

@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json()
    
    if not data or "question" not in data:
        return jsonify({"error": "No question provided"}), 400
    
    question = data["question"]
    chatbot_output = run_chatbot(question)
    
    return jsonify({
        "response": chatbot_output.get("response", "Sorry, I couldn't process your question."),
        "bleu": chatbot_output.get("bleu", None),
        "similarity": chatbot_output.get("similarity", None)
    })

if __name__ == "__main__":
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)