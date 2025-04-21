import argparse
import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))  # Path to data (provided by SageMaker)
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))  # Where to save the model
    return parser.parse_args()

def main():
    args = parse_args()

    # Load your dataset
    dataset_path = os.path.join(args.data, 'healthcare.csv')  # Path to the CSV file in S3
    df = pd.read_csv(dataset_path)
    
    # Preprocess the data: Encoding categorical columns (service_type and location)
    label_encoder_service = LabelEncoder()
    label_encoder_location = LabelEncoder()
    df['service_type'] = label_encoder_service.fit_transform(df['service_type'])
    df['location'] = label_encoder_location.fit_transform(df['location'])
    
    # Prepare the data
    X = df[['service_type', 'location']]  # Features: service_type and location
    y = df['rating']  # Target: rating
    
    # Split the data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    predictions = model.predict(X_val)
    mae = mean_absolute_error(y_val, predictions)
    print(f"Model Mean Absolute Error: {mae:.4f}")
    
    # Save the model
    model_path = os.path.join(args.model_dir, 'model.joblib')
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == '__main__':
    main()
