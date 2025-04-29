# Install necessary packages
!pip install huggingface_hub --upgrade
!pip install docx2txt
!pip install mlflow

from huggingface_hub import InferenceClient
import docx2txt
from google.colab import files
import mlflow
import time
import json
import os

# Initialize Hugging Face Inference Client
client = InferenceClient(api_key="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")

# Set up MLFlow experiment
mlflow.set_experiment("LLMOps_Phi_Model_Tracking")

# Create a folder to save logs
if not os.path.exists("mlflow_logs"):
    os.makedirs("mlflow_logs")

def extract_text_from_docx(docx_path):
    """
    Extracts text from a Word document.

    Parameters:
    docx_path (str): Path to the Word document.

    Returns:
    str: Extracted text from the document.
    """
    return docx2txt.process(docx_path)

def ask_question_with_phi(question, context):
    """
    Asks a question using the 'microsoft/Phi-3.5-mini-instruct' model with optional context.

    Parameters:
    question (str): The question to ask.
    context (str, optional): Additional context from the document.

    Returns:
    str: The model's response.
    """
    with mlflow.start_run():  # Start an MLFlow run
        try:
            start_time = time.time()  # Record start time
            messages = [
      {
        "role": "user",
        "content": f"""Use only the information in the following context to answer the question. If the answer is not found, say 'I could not find that in the document.'
        Context: {context}
        Question: {question}"""
      }
    ]

            # Hugging Face API call
            output = client.chat.completions.create(
                model="microsoft/Phi-3.5-mini-instruct",
                messages=messages,
                stream=True,
                temperature=0.5,
                max_tokens=2048,
                top_p=0.7
            )

            # Capture the full response
            full_response = []
            for chunk in output:
                full_response.append(chunk.choices[0].delta.content)
            response_text = "".join(full_response)

            # Calculate metrics
            latency = time.time() - start_time
            question_token_count = len(question.split())
            context_token_count = len(context.split())
            response_token_count = len(response_text.split())
            response_length = len(response_text)
            estimated_cost = (question_token_count + context_token_count+ response_token_count) * 0.00001  # hypothetical cost per token

            # Log metrics and parameters
            mlflow.log_param("model", "microsoft/Phi-3.5-mini-instruct")
            mlflow.log_param("question", question)
            mlflow.log_metric("latency", latency)
            mlflow.log_metric("question_token_count", question_token_count)
            mlflow.log_metric("context_token_count", context_token_count)
            mlflow.log_metric("response_token_count", response_token_count)
            mlflow.log_metric("response_length", response_length)
            mlflow.log_metric("estimated_cost", estimated_cost)
            mlflow.log_text(response_text, "response_text.txt")

            # Save logs to a local JSON file
            log_data = {
                "model": "microsoft/Phi-3.5-mini-instruct",
                "question": question,
                "latency": latency,
                "question_token_count": question_token_count,
                "context_token_count" : context_token_count,
                "response_token_count": response_token_count,
                "response_length": response_length,
                "estimated_cost": estimated_cost,
                "response_text": response_text
            }


            with open(f"mlflow_logs/log_{time.time()}.json", "w") as log_file:
                json.dump(log_data, log_file)

            return response_text

        except Exception as e:
            mlflow.log_metric("error", 1)  # Log error occurrence
            mlflow.log_text(str(e), "error_message.txt")
            return f"Error: {str(e)}"

def chatbot_with_document(context):
    """
    Simple chatbot that uses 'microsoft/Phi-3.5-mini-instruct' to answer based on the document content.

    Parameters:
    context (str): The context for answering questions.
    """
    print("Hello! I am your chatbot. Ask me anything, or type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        answer = ask_question_with_phi(user_input, context)
        print("Bot:", answer)

if __name__ == "__main__":
    print("Please upload a Word document.")
    uploaded = files.upload()
    if uploaded:
        docx_path = list(uploaded.keys())[0]
        context = extract_text_from_docx(docx_path)
        chatbot_with_document(context)
    else:
        print("No document uploaded. Exiting.")

# Zip the logs for easy download
!zip -r mlflow_logs.zip mlflow_logs/
print("Download the logs zip file for MLFlow metrics:", "/content/mlflow_logs.zip")
