# Task 2 -> Translation
# Input : Medical Report in .pdf format. Uploaded.
# Output : Summary of report, translated from English to Indian Vernacular Languages. 

# Installing dependencies
!pip install openai==1.13.3 httpx==0.27.0 PyMuPDF

# Importing necessary modules
import fitz  # PyMuPDF
from openai import OpenAI
from google.colab import userdata
from google.colab import files

# Uploading file
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

# Translating text using ChatGPT
def translate_text(text, target_language):
    translation_prompt = f"Translate the following text into {target_language}:\n\n{text}"
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant fluent in multiple languages."},
            {"role": "user", "content": translation_prompt}
        ],
        temperature=0.3,
        max_tokens=500
    )
    return response.choices[0].message.content.strip()

# Prompting user for translation option
def prompt_for_translation(summary):
    user_input = input("Do you need the summary to be translated? (yes/no): ").strip().lower()
    if user_input == 'yes':
        print("Please select a language from the following options:")
        print("1. Hindi")
        print("2. Bengali")
        print("3. Tamil")
        print("4. Telugu")
        print("5. Marathi")
        print("6. Gujarati")
        print("7. Kannada")
        print("8. Malayalam")
        print("9. Punjabi")
        language_choice = input("Enter the number of the language you prefer: ").strip()

        language_dict = {
            '1': 'Hindi',
            '2': 'Bengali',
            '3': 'Tamil',
            '4': 'Telugu',
            '5': 'Marathi',
            '6': 'Gujarati',
            '7': 'Kannada',
            '8': 'Malayalam',
            '9': 'Punjabi'
        }

        target_language = language_dict.get(language_choice, 'Hindi')  # Default to Hindi if invalid choice
        print(f"Translating to {target_language}...")
        translated_summary = translate_text(summary, target_language)
        return translated_summary
    else:
        return summary

# Main function
def summarize_report(file_path):
    text = extract_text_from_pdf(file_path)
    summary = summarize_text(text)
    translated_summary = prompt_for_translation(summary)
    return translated_summary

# Generate and display the final summary
report_summary = summarize_report(file_path)

print("=== Final Report Summary ===\n")
print(report_summary)


