import os
import requests
import streamlit as st
from dotenv import load_dotenv
from io import BytesIO  # Import BytesIO for handling image data

# Load environment variables
load_dotenv()

# Fetch API Token from environment variables
API_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# Validate API token
if not API_TOKEN:
    st.error("🚨 API token is missing! Set it in the .env file and restart the app.")
    st.stop()  # Stop execution if no API token is found

# Hugging Face API URL
API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"

# Headers for authentication
HEADERS = {"Authorization": f"Bearer {API_TOKEN}"}

# Streamlit UI
st.title("🎨 AI Image Generator with Stable Diffusion")
st.write("Enter a prompt and generate an AI image!")

# User input for prompt
prompt = st.text_input("Enter your image prompt:", "A futuristic cyberpunk city at sunset, ultra-detailed, 4K")

# Button to generate image
if st.button("Generate Image"):
    if prompt.strip():  # Ensure the prompt is not empty
        with st.spinner("⏳ Generating image... Please wait."):
            try:
                response = requests.post(API_URL, json={"inputs": prompt}, headers=HEADERS, timeout=30)

                if response.status_code == 200:
                    image = BytesIO(response.content)  # Convert response to image
                    st.image(image, caption="🖼️ Generated Image", use_container_width=True)  # Updated parameter
                    st.success("✅ Image generated successfully!")
                else:
                    error_message = response.json().get("error", "Unknown error")
                    st.error(f"❌ API Error ({response.status_code}): {error_message}")

            except requests.exceptions.Timeout:
                st.error("❌ Request timed out. The server took too long to respond.")
            except requests.exceptions.ConnectionError:
                st.error("❌ Network error: Unable to connect to the API.")
            except requests.exceptions.RequestException as e:
                st.error(f"❌ Unexpected error: {e}")

    else:
        st.warning("⚠️ Please enter a valid prompt.")
