import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import boto3
import os

# Parse command-line arguments for input and output paths
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training-data', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    parser.add_argument('--output-dir', type=str, default=os.environ.get('SM_OUTPUT_DIR'))
    return parser.parse_args()

# Load and preprocess the data
def load_and_preprocess_data(s3_data_path):
    # Read CSV from S3
    df = pd.read_csv(s3_data_path)
    
    # Map feedback (ratings) from 5-star scale to binary class (for simplicity)
    df['feedback'] = df['feedback'].apply(lambda x: 1 if x >= 4 else 0)  # 1 if positive, 0 if negative
    
    # Split data into features (X) and target (y)
    X = df[['salary']]  # Using only salary for prediction as an example
    y = df['feedback']
    
    return X, y

# Train the model
def train_model(X, y):
    model = LogisticRegression()
    model.fit(X, y)
    return model

# Save model to output path
def save_model(model, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, 'model.joblib')
    joblib.dump(model, model_path)
    print(f'Model saved at {model_path}')

# Main training loop
def main():
    args = parse_args()
    
    # Load data from the S3 path
    print(f'Loading data from {args.training_data}')
    X, y = load_and_preprocess_data(args.training_data)
    
    # Train the model
    print('Training model...')
    model = train_model(X, y)
    
    # Save the trained model to output
    print('Saving model...')
    save_model(model, args.output_dir)
    
    print('Training complete!')

if __name__ == '__main__':
    main()
