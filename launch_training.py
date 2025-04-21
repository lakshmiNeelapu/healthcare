import sagemaker
from sagemaker.estimator import Estimator
from sagemaker import Session

# AWS config
role = 'arn:aws:iam::522814723070:role/service-role/AmazonSageMaker-ExecutionRole-20250421T200444'
region = 'us-east-1'
sagemaker_session = sagemaker.Session(boto_region_name=region)

# S3 paths
training_data_uri = 's3://healthcaretesthhax/healthcare.csv.txt'
output_model_uri = 's3://healthcaretesthhax/output/'

# Define estimator
estimator = Estimator(
    role=role,
    instance_count=1,
    instance_type='ml.t3.medium',   # ✅ Budget-friendly instance for training
    entry_point='train.py',         # Your training script
    source_dir='.',                 # Path to your train.py file (local)
    output_path=output_model_uri,
    framework_version='1.0',
    base_job_name='healthcare-training-job',
    image_uri='382416733822.dkr.ecr.us-east-1.amazonaws.com/linear-learner:latest'  # Built-in image for us-east-1
)

# Launch training job
estimator.fit({'training': training_data_uri})

