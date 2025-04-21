import sagemaker
from sagemaker import get_execution_role
from sagemaker.sklearn import SKLearn
from sagemaker.inputs import TrainingInput

# Set your S3 paths
bucket_name = 'healthcaretesthhax'

# S3 paths for training data, output, and script
training_data_uri = 'https://healthcaretesthhax.s3.us-east-1.amazonaws.com/healthcare.csv'  # Correct path to your dataset in S3
output_path = f's3://healthcaretesthhax/output/'  # Path to where the model should be saved
script_path = f's3://healthcaretesthhax/src/train.py'  # Path to your train.py script in S3

# Define SageMaker execution role
role = get_execution_role()

# Create the SageMaker SKLearn Estimator
sklearn_estimator = SKLearn(
    entry_point='train.py',  # The training script
    source_dir='src',  # Directory where train.py is located
    role=role,
    instance_count=1,  # Number of instances for training
    instance_type='ml.m5.large',  # Instance type
    framework_version='0.23-1',  # Scikit-learn version
    output_path=output_path,  # S3 output path for model artifacts
    base_job_name="healthcare-rating-prediction"  # Job name
)

# Define the training input data
training_input = TrainingInput(training_data_uri, content_type='csv')

# Start the training job
sklearn_estimator.fit({'training': training_input})
