from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # This will allow CORS for all routes and origins

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    return jsonify({'status': 'success', 'message': 'File uploaded successfully'})

@app.route('/login', methods=['POST'])
def login():
    # Retrieve username and password from the request
    username = request.form.get('username')
    password = request.form.get('password')
    
    # Check if the username and password are provided
    if username and password:
        return jsonify({'status': 'success', 'message': 'Login successful'})
    else:
        return jsonify({'status': 'error', 'message': 'Missing username or password'}), 400

if __name__ == '__main__':
    app.run(debug=True, port=8080, host='0.0.0.0')
