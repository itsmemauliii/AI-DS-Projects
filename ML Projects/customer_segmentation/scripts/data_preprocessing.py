import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(filepath):
    """Load and preprocess customer data"""
    df = pd.read_csv(filepath)
    
    # Select relevant features
    X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
    
    # Standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, df

if __name__ == "__main__":
    X_scaled, df = preprocess_data('../data/Mall_Customers.csv')
    print("Data preprocessing completed!")
