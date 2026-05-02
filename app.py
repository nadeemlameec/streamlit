import streamlit as st
from google import genai
from google.genai import types

st.set_page_config(page_title="IAS & Data Analyst AI", layout="wide")

# Secure API Key handling
api_key = st.secrets.get("GEMINI_API_KEY") or st.sidebar.text_input("Enter API Key", type="password")

if not api_key:
    st.warning("Please provide an API Key to continue.")
    st.stop()

client = genai.Client(api_key=api_key)

st.title("🏛️ IAS & Marketing Analyst AI")

# --- File Upload Section ---
st.sidebar.header("📁 Data Analysis")
uploaded_file = st.sidebar.file_uploader("Upload campaign reports (CSV, PDF, TXT)", type=["csv", "pdf", "txt"])

if uploaded_file:
    # Read the content of the file
    file_content = uploaded_file.read().decode("utf-8") if uploaded_file.type != "application/pdf" else "PDF Content placeholder"
    st.sidebar.success("File uploaded successfully!")
    
    # Button to trigger specific insights
    if st.sidebar.button("Generate Insights from File"):
        with st.spinner("Analyzing data..."):
            # We pass the file content directly into the prompt for the model to analyze
            analysis_prompt = f"Analyze the following campaign data and provide key insights on ROI, incremental sales, and responder behavior:\n\n{file_content}"
            
            response = client.models.generate_content(
                model="gemini-1.5-flash", # Using stable version to avoid ClientErrors
                contents=analysis_prompt,
                config=types.GenerateContentConfig(
                    system_instruction="You are a senior Marketing Analyst. Summarize the data into actionable insights.",
                    temperature=0.2, # Lower temperature for factual accuracy
                ),
            )
            st.markdown("### 📊 File Insights")
            st.write(response.text)
            st.divider()

# --- Standard Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question or discuss the uploaded file..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # If a file is uploaded, we include its context in every chat message
        context_prompt = f"Context from uploaded file:\n{file_content[:2000]}\n\nUser Question: {prompt}" if uploaded_file else prompt
        
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=[context_prompt],
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(include_thoughts=True),
                system_instruction="You are a versatile assistant. If the user asks about uploaded data, act as a Marketing Analyst. Otherwise, act as an IAS Aspirant.",
                temperature=0.5,
            ),
        )
        st.markdown(response.text)
        st.session_state.messages.append({"role": "assistant", "content": response.text})
