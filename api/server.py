from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load model yang telah disimpan
model_path = "model/best_model.pkl"
with open(model_path, "rb") as file:
    model = pickle.load(file)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Ambil data dari permintaan
        data = request.get_json()

        # Konversi data menjadi DataFrame
        input_data = pd.DataFrame([data])

        # Lakukan prediksi
        prediction = model.predict(input_data)

        # Kembalikan hasil prediksi
        return jsonify({"prediction": prediction.tolist()}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
