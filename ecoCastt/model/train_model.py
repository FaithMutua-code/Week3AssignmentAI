import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

def train_emissions_model():
    print("Loading dataset...")
    
    try:
        # Load the dataset with proper parsing
        df = pd.read_csv('data/export_emissions.csv', skiprows=1)
        
        # The first column is years, rest are countries
        years = df.iloc[:, 0].values  # First column contains years
        countries = df.columns[1:]    # Skip the first column (years)
        
        print(f"Loaded data for {len(countries)} countries from {years[0]} to {years[-1]}")
        
        # Prepare data for training
        X = []
        y = []
        country_names = []
        year_values = []
        
        for country_idx, country in enumerate(countries):
            if pd.isna(country):
                continue
                
            emissions_data = df[country].values
            
            for i, year in enumerate(years):
                if (i < len(emissions_data) and 
                    not pd.isna(emissions_data[i]) and 
                    str(emissions_data[i]).strip() != '' and
                    str(emissions_data[i]).strip() != 'nan'):
                    
                    try:
                        emission_value = float(emissions_data[i])
                        
                        # Use previous 3 years as features (if available)
                        features = []
                        valid_history_count = 0
                        
                        for j in range(1, 4):  # Use last 3 years
                            if i - j >= 0 and i - j < len(emissions_data):
                                hist_value = emissions_data[i - j]
                                if not pd.isna(hist_value) and str(hist_value).strip() != '':
                                    features.append(float(hist_value))
                                    valid_history_count += 1
                                else:
                                    features.append(0.0)
                            else:
                                features.append(0.0)
                        
                        # Add year and country encoding as features
                        features.extend([year, country_idx])
                        
                        X.append(features)
                        y.append(emission_value)
                        country_names.append(country)
                        year_values.append(year)
                        
                    except (ValueError, TypeError) as e:
                        continue
        
        if len(X) == 0:
            raise ValueError("No valid training data found!")
            
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        
        print(f"Prepared {len(X)} training samples")
        print(f"Feature shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=True
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")
        
        # Train the model with simpler parameters
        model = RandomForestRegressor(
            n_estimators=50, 
            random_state=42, 
            max_depth=10,
            n_jobs=-1
        )
        
        print("Training model...")
        model.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"✅ Model trained successfully!")
        print(f"📊 Mean Absolute Error: {mae:.4f}")
        print(f"📈 R² Score: {r2:.4f}")
        
        # Save the model
        os.makedirs('model', exist_ok=True)
        joblib.dump(model, 'model/emissions_model.joblib')
        
        # Save country list for reference
        unique_countries = sorted(list(set(country_names)))
        joblib.dump(unique_countries, 'model/countries.joblib')
        
        print(f"💾 Model saved for {len(unique_countries)} countries")
        
        return model, unique_countries
        
    except Exception as e:
        print(f"❌ Error in training: {e}")
        raise

if __name__ == "__main__":
    train_emissions_model()