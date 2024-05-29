from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import os

# Importar funções necessárias do model.py
from model import recebe_imagem, weather_predictor

app = Flask(__name__)

# Configuração para upload de arquivos
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        results = process_image(filepath)
        clima = process_climate(results)
        return render_template('index.html', image_url=url_for('uploaded_file', filename=filename), results=results, clima=clima)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

def process_image(filepath):
    # Processar a imagem com o model.py
    results = recebe_imagem(filepath)
    return results

def process_climate(results):
    # Processar o clima com o model.py
    clima = weather_predictor(results)
    return clima

if __name__ == '__main__':
    app.run(debug=True)
