# Using fine-tuned model

# Installing dependencies
!pip install openai==1.13.3 httpx==0.27.0

# Importing necessary modules
from openai import OpenAI
from google.colab import userdata

# Initialize OpenAI client
client = OpenAI(api_key=userdata.get('OPENAPI_TOKEN'))

# Respond using the fine-tuned model
def respond_to_emergency(query):
    response = client.chat.completions.create(
        model="ft:gpt-3.5-turbo-0125:personal::BQxp4Pz6",  # Fine-tuned model ID
        messages=[
            {"role": "system", "content": "You are a helpful emergency response assistant."},
            {"role": "user", "content": query}
        ],
        temperature=0.5,
        max_tokens=150
    )
    # Extracting the response from the chat model
    return response

# Main function to take user input and get emergency response
def main():
    query = input("Enter your emergency query: ")
    response = respond_to_emergency(query)
    print("Emergency Response:", response.choices[0].message.content)

# Run the main function to interact with the fine-tuned model
if __name__ == "__main__":
    main()
