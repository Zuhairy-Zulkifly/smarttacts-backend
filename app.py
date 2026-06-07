from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# --- CONFIGURATION PATHS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "smarttactsMockv1.pkl")
CSV_PATH = os.path.join(BASE_DIR, "test_dataset_300_1.csv")

# --- DATABASE DRILL MENGIKUT FORMASI ---
DRILL_DATABASE = {
    "2-2": {
        "title": "Square Passing & Interception",
        "description": "Fokus kepada pengekalan struktur bentuk segi empat (2-2) bawah tekanan, serta latihan memintas hantaran tengah lawan."
    },
    "1-3": {
        "title": "Pivot Target Play & Shooting",
        "description": "Latihan menghantar bola terus kepada target man (Pivot). Pemain sayap (Ala) melakukan larian laju untuk rembatan klinikal."
    },
    "3-1": {
        "title": "Counter Attack Pressing",
        "description": "Sistem bertahan rendah yang disiplin (3 pemain di bawah), diikuti transisi serangan pantas sebaik sahaja bola berjaya dirampas."
    },
    "0-4": {
        "title": "All-Out Attacking Overload",
        "description": "Sesuai untuk situasi mengejar gol. Fokus kepada corak hantaran pantas (one-touch) untuk memecahkan blok pertahanan rendah."
    },
    "4-0": {
        "title": "Ultra-Defensive Low Block",
        "description": "Latihan mematikan ruang tengah. Semua pemain menutup ruang dalam kawasan sendiri untuk mengekang serangan bertubi-tubi."
    },
    "2-1-1": {
        "title": "Y-Formation Transition Play",
        "description": "Fokus kepada kestabilan kawasan tengah (Anchor & Midfielder) untuk mengawal tempo perlawanan sebelum menghantar ke sayap."
    }
}

# --- LOAD MODEL ---
try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print("\n" + "="*50)
        print(f"--- [SUCCESS] Model SmartTacts Dimuat dari: {MODEL_PATH}")
        if hasattr(model, 'feature_names_in_'):
            print(f"--- [INFO] Features: {list(model.feature_names_in_)}")
        print("="*50 + "\n")
    else:
        print(f"--- [ERROR] Fail model tidak dijumpai di: {MODEL_PATH} ---")
        model = None
except Exception as e:
    print(f"--- [ERROR] Gagal muat model: {e} ---")
    model = None

# --- ROUTES ---

@app.route('/')
def home():
    return f"<h1>Server SmartTacts sedang berjalan!</h1><p>Status Model: {'Ready' if model else 'Error'}</p>"

@app.route('/predict', methods=['POST'])
def predict():
    print("\n" + "-"*30)
    print("--- REQUEST PREDICTION MASUK ---")
    try:
        data = request.get_json()
        raw_features = data.get('features', [])

        if model is not None and len(raw_features) > 0:
            input_data = np.array(raw_features).reshape(1, -1)
            prediction = model.predict(input_data)
            hasil_ai = str(prediction[0])
            print(f"DEBUG: Success! Hasil Prediction -> {hasil_ai}")

            # Ambil data drill mengikut formasi. Kalau tak jumpa, bagi default drill.
            drill_info = DRILL_DATABASE.get(hasil_ai, {
                "title": "Tactical Shadow Play",
                "description": "Latihan simulasi pergerakan posisi asas pasukan untuk membina keserasian antara pemain."
            })

            return jsonify({
                "status": "success",
                "prediction": hasil_ai,
                "drill_title": drill_info["title"],
                "drill_desc": drill_info["description"]
            })
        else:
            return jsonify({"status": "error", "prediction": "Model/Features Error"}), 500
    except Exception as e:
        print(f"CRITICAL ERROR: {str(e)}")
        return jsonify({"status": "error", "prediction": "NO RESULT", "details": str(e)}), 500

# --- ROUTE UNTUK ADMIN DASHBOARD ---
@app.route('/admin/stats', methods=['GET'])
def get_admin_stats():
    print("\n--- REQUEST ADMIN STATS MASUK ---")
    try:
        if os.path.exists(CSV_PATH):
            df = pd.read_csv(CSV_PATH)

            stats = {
                "total_records": len(df),
                "top_patterns": df['recommended_pattern'].value_counts().head(5).to_dict(),
                "avg_possession": round(df['possession_tendency'].mean(), 1),
                "formation_counts": df['base_formation'].value_counts().to_dict()
            }
            return jsonify({"status": "success", "data": stats})
        else:
            return jsonify({"status": "error", "message": "CSV tidak dijumpai"}), 404
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)