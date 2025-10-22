from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os
import sys

app = Flask(__name__)

# Global variables for model and data
model = None
countries = []
df = None

def load_model_and_data():
    global model, countries, df
    
    try:
        print("Loading model and data...")
        
        # Load model if it exists
        if os.path.exists('model/emissions_model.joblib'):
            model = joblib.load('model/emissions_model.joblib')
            print("✅ Model loaded successfully")
        else:
            print("❌ Model file not found")
            return False
        
        # Load countries list
        if os.path.exists('model/countries.joblib'):
            countries = joblib.load('model/countries.joblib')
            print(f"✅ Loaded {len(countries)} countries")
        else:
            print("❌ Countries file not found")
            return False
        
        # Load historical data
        if os.path.exists('data/export_emissions.csv'):
            df = pd.read_csv('data/export_emissions.csv', skiprows=1)
            print("✅ Historical data loaded")
        else:
            print("❌ Data file not found")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Error loading model or data: {e}")
        return False

def predict_emissions(country, target_year):
    if model is None or df is None:
        return None
    
    try:
        # Find the country in our dataset
        if country in df.columns:
            country_data = df[country].values
            years = df.iloc[:, 0].values  # First column is years
            
            # Find the most recent valid data points
            recent_data = []
            valid_points = 0
            
            # Look backwards for the last 3 years of data
            for i in range(len(country_data)-1, -1, -1):
                if (i >= 0 and not pd.isna(country_data[i]) and 
                    str(country_data[i]).strip() != '' and
                    str(country_data[i]).strip() != 'nan'):
                    try:
                        recent_data.append(float(country_data[i]))
                        valid_points += 1
                        if valid_points >= 3:  # We need 3 data points
                            break
                    except (ValueError, TypeError):
                        continue
            
            # If we don't have enough historical data, pad with zeros
            while len(recent_data) < 3:
                recent_data.append(0.0)
            
            # Get country index
            country_idx = countries.index(country) if country in countries else 0
            
            # Prepare features: [last_3_years_emissions, target_year, country_index]
            features = recent_data[:3]  # Last 3 years of data
            features.extend([target_year, country_idx])
            
            prediction = model.predict([features])[0]
            return max(0, prediction)  # Ensure non-negative prediction
            
        return None
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html', countries=sorted(countries))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        country = data.get('country')
        year = int(data.get('year', 2030))
        
        if not country or country not in countries:
            return jsonify({'error': 'Invalid country'}), 400
        
        if year < 2020 or year > 2050:
            return jsonify({'error': 'Year must be between 2020 and 2050'}), 400
        
        prediction = predict_emissions(country, year)
        
        if prediction is not None:
            return jsonify({
                'country': country,
                'year': year,
                'prediction': round(prediction, 2),
                'unit': 'MtCO₂'
            })
        else:
            return jsonify({'error': 'Could not generate prediction for this country'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/countries')
def get_countries():
    return jsonify(sorted(countries))

@app.route('/health')
def health():
    status = {
        'model_loaded': model is not None,
        'countries_loaded': len(countries) > 0,
        'data_loaded': df is not None,
        'countries_count': len(countries)
    }
    return jsonify(status)

if __name__ == '__main__':
    # Add model directory to path
    sys.path.append('model')
    
    # Train model if it doesn't exist
    if not os.path.exists('model/emissions_model.joblib'):
        print("🚀 Training model for the first time...")
        try:
            from train_model import train_emissions_model
            train_emissions_model()
            print("✅ Model training completed")
        except Exception as e:
            print(f"❌ Model training failed: {e}")
            sys.exit(1)
    
    # Load the trained model
    if load_model_and_data():
        print("🌍 Ecocast is ready!")
        print(f"📊 Available countries: {len(countries)}")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to start Ecocast")
        sys.exit(1)