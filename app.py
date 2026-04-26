from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from werkzeug.utils import secure_filename
import os
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import json
import joblib

app = Flask(__name__)
app.secret_key = 'supersecretkey'
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

MODEL_PATH = 'ad_model.joblib'
METRICS_PATH = 'model_metrics.json'

# Global state for model info
model_info = {
    'status': 'Not Trained',
    'tech': 'N/A',
    'trained_at': 'N/A',
    'samples': 0
}

# Try to load existing model info
if os.path.exists(METRICS_PATH):
    try:
        with open(METRICS_PATH, 'r') as f:
            model_info.update(json.load(f))
    except: pass

# RAPIDS Integration
try:
    import cudf
    import cuml
    from cuml.linear_model import LogisticRegression
    from cuml.metrics import accuracy_score, precision_score, recall_score
    HAS_RAPIDS = True
    print("RAPIDS GPU Acceleration Enabled")
except ImportError:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    HAS_RAPIDS = False
    print("Warning: RAPIDS not found. Using Scikit-Learn CPU.")

def init_sqlite_db():
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            age REAL,
            gender INTEGER,
            time_spent REAL,
            device_type INTEGER,
            ad_category INTEGER,
            ad_position INTEGER,
            link_length INTEGER,
            prediction INTEGER,
            probability REAL,
            explanation TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Simple migration logic for missing columns
    cursor.execute("PRAGMA table_info(history)")
    columns = [col[1] for col in cursor.fetchall()]
    if 'ad_category' not in columns:
        cursor.execute("ALTER TABLE history ADD COLUMN ad_category INTEGER")
    if 'ad_position' not in columns:
        cursor.execute("ALTER TABLE history ADD COLUMN ad_position INTEGER")
    if 'probability' not in columns:
        cursor.execute("ALTER TABLE history ADD COLUMN probability REAL")
    if 'explanation' not in columns:
        cursor.execute("ALTER TABLE history ADD COLUMN explanation TEXT")
    
    conn.commit()
    conn.close()

init_sqlite_db()

def train_model(filepath):
    global model_info
    if HAS_RAPIDS:
        df = cudf.read_csv(filepath)
        X = df.drop('clicked', axis=1)
        y = df['clicked']
        model = LogisticRegression()
        model.fit(X, y)
        preds = model.predict(X)
        acc = accuracy_score(y, preds)
        # cuML metrics need conversion to host if using scikit-learn style or just use cuml.metrics
        metrics = {
            'accuracy': float(acc),
            'samples': len(df),
            'tech': 'RAPIDS GPU',
            'trained_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        # Get coefficients for importance (conversion to pandas for ease)
        coefs = model.coef_.to_pandas().values[0]
    else:
        df = pd.read_csv(filepath)
        X = df.drop('clicked', axis=1)
        y = df['clicked']
        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)
        preds = model.predict(X)
        acc = accuracy_score(y, preds)
        metrics = {
            'accuracy': float(acc),
            'samples': len(df),
            'tech': 'Scikit-Learn CPU',
            'trained_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        coefs = model.coef_[0]

    # Save model and metrics
    joblib.dump(model, MODEL_PATH)
    
    # Feature importance labels
    features = ['age', 'gender', 'time_spent', 'device_type', 'ad_category', 'ad_position', 'link_length']
    importance = {feat: float(abs(coef)) for feat, coef in zip(features, coefs)}
    metrics['importance'] = importance
    metrics['status'] = 'Active'
    
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f)
    
    model_info.update(metrics)
    return metrics

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        data = request.json
        username = data.get('username')
        password = data.get('password')
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username=?", (username,))
        user = cursor.fetchone()
        if not user:
            cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
            conn.commit()
            cursor.execute("SELECT * FROM users WHERE username=?", (username,))
            user = cursor.fetchone()
        
        if user[2] == password:
            session['user_id'] = user[0]
            session['username'] = user[1]
            conn.close()
            return jsonify({'success': True})
        conn.close()
        return jsonify({'success': False, 'message': 'Invalid credentials'})
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if 'user_id' not in session: return redirect(url_for('login'))
    if request.method == 'POST':
        if 'file' not in request.files: return jsonify({'success': False})
        file = request.files['file']
        if file.filename == '': return jsonify({'success': False})
        filename = secure_filename(file.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(path)
        metrics = train_model(path)
        return jsonify({'success': True, 'metrics': metrics})
    return render_template('upload.html')

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session: return redirect(url_for('login'))
    return render_template('dashboard.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'user_id' not in session: return redirect(url_for('login'))
    if request.method == 'POST':
        data = request.json
        if not os.path.exists(MODEL_PATH):
            return jsonify({'success': False, 'message': 'Model not trained yet.'})
        
        model = joblib.load(MODEL_PATH)
        features = ['age', 'gender', 'time_spent', 'device_type', 'ad_category', 'ad_position', 'link_length']
        input_data = pd.DataFrame([[float(data[f]) for f in features]], columns=features)
        
        # RAPIDS prediction needs cudf if trained on cudf, but joblib might have handled it if it's sklearn model
        # For simplicity, we use pandas since we are predicting one row
        prob = model.predict_proba(input_data)[0][1]
        pred = 1 if prob > 0.5 else 0
        
        # Enhanced Explanation Logic (Explainable AI)
        reasons = []
        if float(data['time_spent']) > 40: reasons.append("exceptional session engagement")
        elif float(data['time_spent']) > 20: reasons.append("strong user interest")
        
        if float(data['age']) < 30: reasons.append("high-conversion age demographic")
        
        if int(data['device_type']) == 0: reasons.append("mobile-optimized ad delivery")
        
        if int(data['ad_position']) == 0: reasons.append("premium high-visibility placement")
        
        if float(prob) > 0.7:
            explanation = f"High probability due to {', '.join(reasons) if reasons else 'consistent behavioral patterns'}."
        elif float(prob) > 0.4:
            explanation = f"Moderate interest detected based on {reasons[0] if reasons else 'general user profile'}."
        else:
            explanation = "Low click probability; user demographics and placement do not align with current campaign targets."
        
        # Save to history
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO history (user_id, age, gender, time_spent, device_type, ad_category, ad_position, link_length, prediction, probability, explanation)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (session['user_id'], data['age'], data['gender'], data['time_spent'], data['device_type'], data['ad_category'], data['ad_position'], data['link_length'], pred, float(prob), explanation))
        conn.commit()
        conn.close()
        
        return jsonify({
            'success': True,
            'prediction': 'Click' if pred == 1 else 'No Click',
            'probability': round(prob * 100, 2),
            'explanation': explanation
        })
    return render_template('predict.html')

@app.route('/api/model_info')
def api_model_info():
    return jsonify(model_info)

@app.route('/api/history')
def api_history():
    if 'user_id' not in session: return jsonify([])
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute("SELECT age, gender, time_spent, device_type, prediction, probability, timestamp FROM history WHERE user_id=? ORDER BY timestamp DESC LIMIT 10", (session['user_id'],))
    rows = cursor.fetchall()
    conn.close()
    history = []
    for r in rows:
        history.append({
            'age': r[0], 'gender': 'Male' if r[1] == 1 else 'Female',
            'time': f"{r[2]:.1f}m", 'device': ['Mobile', 'Desktop', 'Tablet'][int(r[3])],
            'result': 'Click' if r[4] == 1 else 'No Click',
            'prob': f"{r[5]*100:.1f}%", 'date': r[6]
        })
    return jsonify(history)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
