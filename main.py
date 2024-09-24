from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import threading

print(tf.__version__)
# Initialize the Flask app
app = Flask(__name__)

# Load the saved LSTM model
try:
    model = load_model('lstms_model.keras')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

# Load and preprocess the dataset
def load_and_preprocess_data(file_path):
    try:
        df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
        df.index.freq = 'MS'
        
        train = df.iloc[:156]
        scaler = MinMaxScaler()
        scaler.fit(train)
        
        print("Data loaded and preprocessed successfully.")
        return df, scaler
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

# df, scaler = load_and_preprocess_data('monthly_milk_production.csv')


# Define the home route
@app.route('/')
def home():
    print("Home route accessed")
    return jsonify({'prediction': "Welcome to the LSTM Prediction API!"})


# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    print("Predict route accessed")
    if 'file' not in request.files:
        return 'No file part', 400
    
    file = request.files['file']
    try:
        df, scaler = load_and_preprocess_data(file)

        n_input = 12
        last_batch = df[-n_input:].values
        
        last_batch_scaled = scaler.transform(last_batch).reshape((1, n_input, 1))
        prediction = model.predict(last_batch_scaled)
        prediction_real = scaler.inverse_transform(prediction)[0][0]
        
        print(f"Prediction made: {prediction_real}")
        return jsonify({'prediction': float(prediction_real)})
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': str(e)})

# Function to run the Flask app
def run_app():
    print("Starting Flask app...")
    app.run(port=5000, use_reloader=False,debug=True)

# Start Flask in a separate thread
if __name__ == '__main__':
    # Start the thread
    flask_thread = threading.Thread(target=run_app)
    flask_thread.start()
    
    # Wait for the Flask thread to finish before the program exits
    flask_thread.join()
