import os
import mimetypes
import numpy as np
from google import genai
from google.genai import types
import base64
import streamlit as st
from PIL import Image
from io import BytesIO


os.environ["GEMINI_API_KEY"] = "AIzaSyDI--JtInPfX5c2licf-vMMOD8NEe6u27g"
# st.secrets["GEMINI_API_KEY"]



def base64_to_numpy_image(base64_data):
    # Case 1: If already string, decode it
    if isinstance(base64_data, str):
        image_bytes = base64.b64decode(base64_data)

    # Case 2: If it's already bytes (raw base64 string)
    elif isinstance(base64_data, bytes):
        print("Data in bytes")
        image_bytes = base64_data  # No decoding needed

    else:
        raise ValueError("Unsupported base64 data format")

    # Load image using PIL
    image = Image.open(BytesIO(image_bytes)).convert("RGB")

    # Convert to NumPy array
    np_array = np.array(image)

    return np_array


def generate(user_query):
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    model = "gemini-2.0-flash-preview-image-generation"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=user_query),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        response_modalities=[
            "IMAGE",
            "TEXT",
        ],
        response_mime_type="text/plain",
    )

    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        if (
            chunk.candidates is None
            or chunk.candidates[0].content is None
            or chunk.candidates[0].content.parts is None
        ):
            continue
        if chunk.candidates[0].content.parts[0].inline_data:
            inline_data = chunk.candidates[0].content.parts[0].inline_data
            data_buffer = inline_data.data
            file_extension = mimetypes.guess_extension(inline_data.mime_type)
            return base64_to_numpy_image(data_buffer)
        else:
            pass


    
st.set_page_config(
    page_title="Image Generator",
    page_icon="🎨",
    layout="centered",
)

# Inject some CSS for custom styles
st.markdown(
    """
    <style>
    .centered-title {
        text-align: center;
        font-size: 3em;
        font-weight: bold;
        color: #4A90E2;
        margin-bottom: 20px;
    }

    .prompt-box input {
        font-size: 1.1em;
        padding: 10px;
        border-radius: 10px;
    }

    .stButton>button {
        background-color: #4A90E2;
        color: white;
        border: None;
        border-radius: 10px;
        padding: 0.6em 2em;
        font-size: 1.1em;
        margin-top: 10px;
    }

    .stButton>button:hover {
        background-color: #357ABD;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

def main():
    st.markdown('<div class="centered-title">🖼️ Image Generator</div>', unsafe_allow_html=True)

    prompt = st.text_input("Enter a prompt to generate an image:", key="prompt", placeholder="e.g., A futuristic city at sunset 🌇")

    if st.button("Generate Image"):
        if not prompt.strip():
            st.warning("⚠️ Please enter a prompt!")
            return

        with st.spinner("🎨 Generating image..."):
            image_np = generate(prompt)  # This should return a NumPy array

        if image_np is not None:
            st.image(image_np, caption="Here is your image!", use_container_width=True)
        else:
            st.error("❌ Failed to generate image. Please try again later.")

if __name__ == "__main__":
    main()