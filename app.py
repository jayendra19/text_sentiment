from flask import Flask,request,render_template,jsonify
from src.pipeline.predict import classify_text


app=Flask(__name__)

from flask import Flask, request, render_template
from src.pipeline.predict import classify_text

app = Flask(__name__)

@app.route('/text', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        text = request.form.get('text')
        results = classify_text(text)
        return render_template('home.html', text=text, results=results)
    return render_template('home.html')








@app.route('/textapi', methods=['POST'])
def text_api():
    if request.method == 'POST':
        data = request.get_json()

        if 'text' in data:
            text = data['text']
            result = classify_text(text)
            return jsonify(result)
        else:
            return jsonify({'error': 'Text not provided in the request'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)


