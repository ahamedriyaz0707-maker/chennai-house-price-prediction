from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib

app = Flask(__name__)
CORS(app)

# Load ML model
model = joblib.load('house_price_model.pkl')
model_columns = joblib.load('model_columns.pkl')
category_info = joblib.load('category_info.pkl')

@app.route('/')
def home():
    return "Chennai House Price Prediction API is Running!"

@app.route('/predict', methods=['POST'])
def predict():

    try:
        data = request.get_json()

        input_data = pd.DataFrame([{
            'area': float(data['area']),
            'bhk': int(data['bhk']),
            'bathroom': int(data['bathroom']),
            'age': float(data['age']),
            'location': data['location'],
            'status': data['status'],
            'builder': data['builder']
        }])

        # Convert categorical values
        for col, categories in category_info.items():
            input_data[col] = pd.Categorical(
                input_data[col],
                categories=categories
            )

        input_encoded = pd.get_dummies(
            input_data,
            drop_first=True
        )

        # Match training features
        input_encoded = input_encoded.reindex(
            columns=model_columns,
            fill_value=0
        )

        prediction = model.predict(input_encoded)[0]

        return jsonify({
            'success': True,
            'predicted_price': round(float(prediction), 2)
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
