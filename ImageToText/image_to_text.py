# Task 4 -> Image-To-Text
# Input : Medical Prescription as image. 
# Output : Generate text based, medicines schedule.


# Installing OpenAI client
!pip install openai==1.13.3 httpx

# Imports
import base64
from openai import OpenAI
from google.colab import userdata, files

# Uploading the image
uploaded = files.upload()
file_path = list(uploaded.keys())[0]

# Initializing OpenAI client
client = OpenAI(api_key=userdata.get("OPENAPI_TOKEN"))

# Function to process the prescription image
def process_prescription_image_with_gpt4(image_path):
    # Read and encode the image in base64
    with open(image_path, "rb") as img_file:
        encoded_image = base64.b64encode(img_file.read()).decode("utf-8")

    # Creating data URL for the image
    mime_type = "image/png" if image_path.lower().endswith(".png") else "image/jpeg"
    image_url = f"data:{mime_type};base64,{encoded_image}"

    # Sending request to gpt-4-turbo with image
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful medical assistant."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """Extract the medicine schedule from this prescription and format it like this:

Morning
Before Food
- Tab demo medicine 1, 1 tablet
- Cap demo medicine 2, 1 tablet
After Food
- Tab demo medicine 3, 1 tablet
- Tab demo medicine 4, 1/2 tablet

Afternoon
After Food
- Tab demo medicine 3, 1 tablet

Night
After Food
- Tab demo medicine 3, 1 tablet
- Tab demo medicine 4, 1/2 tablet

Use similar formatting even if the medicine names are different."""
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url}
                    }
                ]
            }
        ],
        max_tokens=800
    )

    print("\nFormatted Prescription Instructions:\n")
    print(response.choices[0].message.content.strip())


process_prescription_image_with_gpt4(file_path)
