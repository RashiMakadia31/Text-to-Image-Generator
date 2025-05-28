import streamlit as st
from image_generator import generate_image
from llama_cpp_wrapper import enhance_prompt

st.set_page_config(page_title="🖼️ Text-to-Image Generator")

st.title("🖼️ Text-to-Image Generator")
prompt = st.text_input("Enter your prompt:")

use_llm = st.checkbox("Enhance prompt with LLaMA.cpp")

if st.button("Generate Image"):
    final_prompt = enhance_prompt(prompt) if use_llm else prompt
    st.write("**Final Prompt:**", final_prompt)
    image = generate_image(final_prompt)
    st.image(image, caption="Generated Image")
    with open("output.png", "rb") as img_file:
        st.download_button("📥 Download Image", data=img_file, file_name="generated.png")
