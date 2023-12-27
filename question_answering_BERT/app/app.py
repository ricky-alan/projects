from flask import Flask, request, render_template
from transformers import pipeline

app = Flask(__name__)

model = pipeline("question-answering", "rickyalan/indobert-finetuned-indosquad", framework="tf")

context = """Nama saya Ricky Alan, saya tinggal di Batam, Indonesia. Saya adalah lulusan prodi Ilmu Komputer dari Universitas Sumatera Utara. Saya memiliki pemahaman mendalam tentang pemrograman, data analisis dan machine learning. Saya tertarik untuk menerapkan keahlian saya dan memberikan kontribusi positif di bidang data science."""
question = "Dimana saya tinggal?"

@app.route('/')
def home():
    return render_template('index.html', context=context, question=question)

@app.route('/answer', methods=['POST'])
def answer():
    context = request.form['context']
    question = request.form['question']

    result = model(context=context, question=question)
    return render_template('index.html', context=context, question=question, answer=result['answer'], score=result['score'])

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)