from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import os
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np

# Importar funções necessárias do model.py
from model import recebe_imagem, weather_predictor

app = Flask(__name__)

# Configuração para upload de arquivos
UPLOAD_FOLDER = 'uploads'
HEATMAP_FOLDER = 'heatmaps'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['HEATMAP_FOLDER'] = HEATMAP_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists(HEATMAP_FOLDER):
    os.makedirs(HEATMAP_FOLDER)

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
        media, clima = process_climate(results)
        heatmap_filename = generate_heatmap(media)
        return render_template('index.html', image_url=url_for('uploaded_file', filename=filename), 
                               results=results, clima=clima, heatmap_url=url_for('heatmap_file', filename=heatmap_filename))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/heatmaps/<filename>')
def heatmap_file(filename):
    return send_from_directory(app.config['HEATMAP_FOLDER'], filename)

def process_image(filepath):
    # Processar a imagem com o model.py
    results = recebe_imagem(filepath)
    return results

def process_climate(results):
    # Processar o clima com o model.py
    media, clima = weather_predictor(results)
    return media, clima

def generate_heatmap(value):

    if not 0 <= value <= 1:
        raise ValueError("O valor deve estar entre 0 e 1")

    gradient = np.linspace(0, 1, 1000).reshape(1, -1)
    
    fig, ax = plt.subplots(figsize=(8, 2), dpi=100) 
    ax.imshow(gradient, aspect='auto', cmap='coolwarm')
    
    ax.axvline(x=value*1000, color='black', linewidth=2)
    
    ax.set_xlim(0, 1000)
    ax.set_xticks(np.linspace(0, 1000, 11))
    ax.set_xticklabels(np.linspace(0, 1, 11), fontsize=10, weight='bold') 
    ax.set_yticks([])
    
    for spine in ax.spines.values():
        spine.set_edgecolor('gray')
        spine.set_linewidth(1.5)
    
    plt.tight_layout(pad=0.5) 
    
    # Salva a imagem
    heatmap_filename = f'heatmap_{value:.2f}.png'
    heatmap_filepath = os.path.join(app.config['HEATMAP_FOLDER'], heatmap_filename)
    plt.savefig(heatmap_filepath, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    
    return heatmap_filename

if __name__ == '__main__':
    app.run(debug=True)
