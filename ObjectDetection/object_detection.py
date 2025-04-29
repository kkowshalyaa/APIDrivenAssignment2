import openai
import base64
from google.colab import userdata, files

# Set API key
openai.api_key = userdata.get("OPENAPI_TOKEN")

# Encode a local JPG image to base64
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

# Analyze image with GPT-4 Vision
def analyze_xray(image_path, patient_name="John Doe", exam_date="2025-04-23", prompt=None):
    base64_image = encode_image_to_base64(image_path)

    custom_prompt = prompt or f"""
You are a radiologist. Generate a chest X-ray report for the following patient.

Patient Name: {patient_name}
Date of Exam: {exam_date}

Describe all abnormalities if present. Include a structured report.
    """

    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": custom_prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }}
                ]
            }
        ],
        max_tokens=800,
        temperature=0.5
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    image_path = "xray.jpg"
    report = analyze_xray("xray.jpg", patient_name="Alice Smith", exam_date="2025-04-22")
    print("\n📋 Radiology Report:\n")
    print(report)
