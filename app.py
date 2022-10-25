import os
from translator import Translator
from flask import Flask, request, send_from_directory, redirect, url_for

translators = {}
for model_name in os.listdir('models/'):
    print(f'loading: {model_name}')
    translators[model_name] = Translator(model_name)

app = Flask(__name__)

def create_reponse(result, type=True):
    return {
        'status': type,
        ('result' if type else 'error'): result
    }

@app.route('/')
def root():
    return redirect('/index.html')

@app.route('/<path:path>')
def public(path):
    return send_from_directory('public', path)

@app.route('/api/models')
def models():
    models = os.listdir('models/')
    return create_reponse(models)

@app.route('/api/translate')
def translate():
    try:
        args = request.args
        model_name = args.get('model')
        sentence = args.get('sentence')
        translator = translators[model_name]
        res = translator(sentence)
        return create_reponse(res)
    except Exception as e:
        return create_reponse(str(e), type=False)

if __name__ == "__main__":
    app.run(debug=False)

