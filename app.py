import streamlit as st
from google import genai
from google.genai import types

# Setup Page
st.set_page_config(page_title="Ask me Anything", page_icon="🏛️")

# Fetch API Key securely from Streamlit Secrets
# If not found in secrets, fallback to a sidebar input
api_key = st.secrets.get("GEMINI_API_KEY") or st.sidebar.text_input("Enter API Key", type="password")

if not api_key:
    st.warning("Please provide an API Key to continue.")
    st.stop()

client = genai.Client(api_key=api_key)

st.title("🏛️ IAS Aspirant AI")
st.info("Analytical insights for UPSC preparation.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question on Governance or Ethics..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(include_thoughts=True),
                temperature=1.0,
            ),
        )
        st.markdown(response.text)
        st.session_state.messages.append({"role": "assistant", "content": response.text})
