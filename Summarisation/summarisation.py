# Task 1 -> Summarisation
# Input : Medical Report in .pdf format. Uploaded.
# Output : Summary of report.

# Installing dependencies
!pip install openai==1.13.3 httpx==0.27.0 PyMuPDF

# Importing modules
import fitz  # PyMuPDF
from openai import OpenAI
from google.colab import userdata
from google.colab import files

# Upload image file
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
        max_tokens=300
    )
    return response.choices[0].message.content.strip()

# Main function
def summarize_report(file_path):
    text = extract_text_from_pdf(file_path)
    summary = summarize_text(text)
    return summary

report_summary = summarize_report(file_path)

print("=== Medical Report Summary ===\n")
print(report_summary)
