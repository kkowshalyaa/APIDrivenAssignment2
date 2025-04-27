# Creating Fine-Tuning Job

# Installing the OpenAI library
!pip install openai==1.0.0  # Ensure you are using the correct version

# Importing necessary libraries
from openai import OpenAI
from google.colab import userdata, files
import time

# Uploading the dataset file for fine-tuning 
client = OpenAI(api_key=userdata.get("OPENAPI_TOKEN"))  
uploaded = files.upload()


# Getting the file path from the uploaded file
file_path = next(iter(uploaded))  


# Uploading the file to OpenAI for fine-tuning
def upload_file(file_path):
    with open(file_path, "rb") as file:
        response = client.files.create(
            file=file,
            purpose='fine-tune' 
        )
        return response.id

file_id = upload_file(file_path)
print(f"File uploaded successfully. File ID: {file_id}")


# Fine-tuning the model with the uploaded file
def fine_tune_model(file_id):
    # Trigger fine-tuning using the file ID
    response = client.fine_tuning.jobs.create(
        training_file=file_id,  
        model="gpt-3.5-turbo"  
    )
    return response.id

fine_tune_id = fine_tune_model(file_id)
print(f"Fine-tuning job started. Fine-tune ID: {fine_tune_id}")


# Monitor the fine-tuning status
def monitor_fine_tuning(fine_tune_id):
      time.sleep(600)
      response = client.fine_tuning.jobs.list_events(fine_tuning_job_id=fine_tune_id, limit=2)
      print(response)

monitor_fine_tuning(fine_tune_id)
