# Task 3 -> Text Generation
# Input : Medical Report in .pdf format. Uploaded.
# Output : Generate text based, using summary of the report as the context. 

# Installing dependencies
!pip install openai==1.13.3 httpx==0.27.0 PyMuPDF

# Importing necessary modules
import fitz  # PyMuPDF
from openai import OpenAI
from google.colab import userdata
from google.colab import files

# Uploading the file
uploaded = files.upload()
file_path = list(uploaded.keys())[0]

# Initializing OpenAI client
client = OpenAI(api_key=userdata.get('OPENAPI_TOKEN'))

# Extracting text from a PDF
def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Summarizing text using ChatGPT
def summarize_text(text):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful medical assistant."},
            {"role": "user", "content": f"Summarize this medical report:\n\n{text}"}
        ],
        temperature=0.3,
        max_tokens=500
    )
    return response.choices[0].message.content.strip()

# Answering questions using the summary as context
def answer_medical_question(summary, question):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a medical assistant, able to interpret medical reports and provide answers based on available data."},
            {"role": "user", "content": f"Based on the following medical report summary, answer the question: '{question}'\n\nSummary:\n{summary}"}
        ],
        temperature=0.3,
        max_tokens=500
    )
    return response.choices[0].message.content.strip()

# Main function to process the report and handle user questions
def summarize_report_and_answer_questions(file_path):
    # Extracting text from the PDF
    text = extract_text_from_pdf(file_path)
    
    # Summarizing the medical report
    summary = summarize_text(text)
    
    # Example interactive question loop
    while True:
        print("\nAsk a medical question based on the report, or type 'exit' to quit:")
        question = input("Question: ").strip()
        
        if question.lower() == 'exit':
            print("Exiting the medical assistant...")
            break
        
        # Get the answer based on the report summary
        answer = answer_medical_question(summary, question)
        
        print("\nAnswer: ", answer)

# Run the process
summarize_report_and_answer_questions(file_path)

