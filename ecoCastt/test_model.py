import pandas as pd
import numpy as np

def test_data_loading():
    """Test if we can load the data properly"""
    try:
        df = pd.read_csv('data/export_emissions.csv', skiprows=1)
        print("✅ Data loaded successfully")
        print(f"Shape: {df.shape}")
        print(f"Columns: {len(df.columns)}")
        print(f"First few columns: {df.columns[:5].tolist()}")
        print(f"Years: {df.iloc[:5, 0].tolist()}")
        
        # Test a specific country
        test_country = 'United States of America'
        if test_country in df.columns:
            us_data = df[test_country].values
            valid_data = [x for x in us_data if not pd.isna(x) and str(x).strip() != '']
            print(f"✅ {test_country}: {len(valid_data)} valid data points")
        else:
            print(f"❌ {test_country} not found in columns")
            
    except Exception as e:
        print(f"❌ Error loading data: {e}")

if __name__ == "__main__":
    test_data_loading()