import os
import requests
import streamlit as st
import cv2
import numpy as np
from dotenv import load_dotenv
from io import BytesIO
from PIL import Image
import time  

# Load environment variables
load_dotenv()

# Fetch API Token from environment variables
API_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# Validate API token
if not API_TOKEN:
    st.error("🚨 API token is missing! Set it in the .env file and restart the app.")
    st.stop()

# Hugging Face API URL
API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"

# Headers for authentication
HEADERS = {"Authorization": f"Bearer {API_TOKEN}"}

# Ensure 'generated_images' directory exists
OUTPUT_DIR = "generated_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Streamlit UI
st.title("🎨 AI Image Generator with Stable Diffusion")
st.write("Enter a prompt and generate AI images!")

# User input for prompt
prompt = st.text_input("Enter your image prompt:", "A futuristic cyberpunk city at sunset, ultra-detailed, 4K")

# Number of images to generate
num_images = st.slider("Select number of images", min_value=1, max_value=10, value=5)

# Variations to make images different
variations = ["different lighting", "alternate angle", "artistic style", "abstract version", "photorealistic version"]

# Button to generate images
if st.button("Generate Images"):
    if prompt.strip():  # Ensure the prompt is not empty
        with st.spinner("⏳ Generating images... Please wait."):

            images = []
            file_paths = []
            try:
                for i in range(num_images):
                    modified_prompt = f"{prompt}, {variations[i % len(variations)]}"
                    response = requests.post(API_URL, json={"inputs": modified_prompt}, headers=HEADERS, timeout=30)

                    if response.status_code == 200:
                        # Convert response to image
                        image = Image.open(BytesIO(response.content))
                        images.append(image)

                        # Define file path
                        file_name = f"{OUTPUT_DIR}/{prompt.replace(' ', '_')}_{i+1}.png"
                        image.save(file_name, format="PNG")  # Save image as PNG
                        file_paths.append(file_name)

                        # Display the generated image
                        st.image(image, caption=f"🖼️ Generated Image {i+1} ({variations[i % len(variations)]})", use_container_width=True)
                        st.success(f"✅ Image {i+1} generated and saved successfully!")
                    else:
                        error_message = response.json().get("error", "Unknown error")
                        st.error(f"❌ API Error ({response.status_code}): {error_message}")
                        break

                # Create an MP4 animation where each image is displayed for 5 seconds
                if len(images) > 1:
                    video_path = f"{OUTPUT_DIR}/{prompt.replace(' ', '_')}.mp4"
                    frame_size = images[0].size
                    frame_width, frame_height = frame_size
                    
                    # Use 'mp4v' codec for Chrome compatibility
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  

                    fps = 10  # Higher FPS for smoother video
                    seconds_per_image = 5
                    frames_per_image = fps * seconds_per_image

                    video = cv2.VideoWriter(video_path, fourcc, fps, (frame_width, frame_height))

                    for img in images:
                        frame = np.array(img.convert("RGB"))
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        for _ in range(frames_per_image):  # Repeat frame for 5 seconds
                            video.write(frame)

                    video.release()  # Ensure file is closed before playing

                    # Ensure file is fully written before playing
                    time.sleep(1)

                    st.video(video_path)
                    with open(video_path, "rb") as video_file:
                        st.download_button("📥 Download Animation (MP4)", data=video_file, file_name="animation.mp4", mime="video/mp4")

            except requests.exceptions.Timeout:
                st.error("❌ Request timed out. The server took too long to respond.")
            except requests.exceptions.ConnectionError:
                st.error("❌ Network error: Unable to connect to the API.")
            except requests.exceptions.RequestException as e:
                st.error(f"❌ Unexpected error: {e}")
    else:
        st.warning("⚠️ Please enter a valid prompt.")
