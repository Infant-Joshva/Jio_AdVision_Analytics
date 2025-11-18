import boto3
from pathlib import Path

# Your bucket name
BUCKET_NAME = "jioadvision-uploads"
s3 = boto3.client("s3")

# Create a small test file
test_file = Path("test_upload.txt")
test_file.write_text("Hello from Jio AdVision Project!")

# Upload to S3
s3.upload_file(str(test_file), BUCKET_NAME, "test_upload.txt")

print(f"✅ File uploaded successfully to s3://{BUCKET_NAME}/test_upload.txt")
