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
file_path = next(iter(uploaded))  # Get the file name (assuming only one file uploaded)


# Uploading the file to OpenAI for fine-tuning
def upload_file(file_path):
    # Open the file in binary mode and upload it for fine-tuning purposes
    with open(file_path, "rb") as file:
        response = client.files.create(
            file=file,
            purpose='fine-tune'  # Indicate that the file is for fine-tuning
        )
        return response.id

file_id = upload_file(file_path)
print(f"File uploaded successfully. File ID: {file_id}")


# Fine-tuning the model with the uploaded file
def fine_tune_model(file_id):
    # Trigger fine-tuning using the file ID
    response = client.fine_tuning.jobs.create(
        training_file=file_id,  # Pass the file ID for training
        model="gpt-3.5-turbo"  # You can use other models like "gpt-3.5-turbo" based on availability
    )
    return response.id

fine_tune_id = fine_tune_model(file_id)
print(f"Fine-tuning job started. Fine-tune ID: {fine_tune_id}")


# Monitor the fine-tuning status
def monitor_fine_tuning(fine_tune_id):
      # Retrieve the fine-tuning job status
      time.sleep(30)
      #response = client.fine_tuning.jobs.list_events(fine_tuning_job_id=ftevent-Fvw4sCQoshcjqBZXl4P0TQLx, limit=2) 
      response = client.fine_tuning.jobs.list_events(fine_tuning_job_id=fine_tune_id, limit=2)
      print(response)

monitor_fine_tuning(fine_tune_id)
