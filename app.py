import streamlit as st
import pandas as pd
from google import genai
from google.genai import types
import PyPDF2

# ---------------- CONFIG ---------------- #
st.set_page_config(page_title="Marketing Copilot AI", layout="wide")

# ---------------- API KEY ---------------- #
api_key = st.secrets.get("GEMINI_API_KEY") or st.sidebar.text_input("Enter API Key", type="password")

if not api_key:
    st.warning("Please provide an API Key to continue.")
    st.stop()

client = genai.Client(api_key=api_key)

# ---------------- TITLE ---------------- #
st.title("📊 Marketing Copilot AI")

# ---------------- SESSION STATE ---------------- #
if "messages" not in st.session_state:
    st.session_state.messages = []

if "df_insights" not in st.session_state:
    st.session_state.df_insights = None

# ---------------- FILE PROCESSING ---------------- #
def extract_file_content(uploaded_file):
    if uploaded_file.type == "application/pdf":
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text
    else:
        uploaded_file.seek(0)
        return uploaded_file.read().decode("utf-8")


def detect_column(df, possible_names):
    for col in df.columns:
        for name in possible_names:
            if name in col:
                return col
    return None


def process_campaign_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df.columns = [col.strip().lower() for col in df.columns]

    insights = {}

    spend_col = detect_column(df, ["spend", "cost"])
    revenue_col = detect_column(df, ["revenue", "sales"])
    customer_col = detect_column(df, ["customer"])
    channel_col = detect_column(df, ["channel"])
    month_col = detect_column(df, ["month", "date"])

    try:
        if spend_col:
            insights["total_spend"] = float(df[spend_col].sum())
        if revenue_col:
            insights["total_revenue"] = float(df[revenue_col].sum())

        if spend_col and revenue_col:
            insights["roi"] = insights["total_revenue"] / insights["total_spend"]

        if customer_col:
            insights["total_customers"] = int(df[customer_col].sum())

        if channel_col and revenue_col:
            channel_perf = (
                df.groupby(channel_col)[revenue_col]
                .sum()
                .sort_values(ascending=False)
            )
            insights["top_channel"] = channel_perf.index[0]
            insights["channel_breakdown"] = channel_perf.to_dict()

        if month_col and revenue_col:
            monthly = df.groupby(month_col)[revenue_col].sum()
            insights["monthly_trend"] = monthly.to_dict()

    except Exception as e:
        insights["error"] = str(e)

    return insights


# ---------------- SIDEBAR ---------------- #
st.sidebar.header("📁 Upload Data")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV or PDF", type=["csv", "pdf", "txt"]
)

file_content = ""

if uploaded_file:
    file_content = extract_file_content(uploaded_file)
    st.sidebar.success("File uploaded successfully!")

    # Process CSV into structured insights
    if uploaded_file.type == "text/csv":
        uploaded_file.seek(0)
        st.session_state.df_insights = process_campaign_data(uploaded_file)

# ---------------- KPI DISPLAY ---------------- #
if st.session_state.df_insights:
    insights = st.session_state.df_insights

    st.subheader("📈 Key Metrics")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Spend", f"₹{insights.get('total_spend', 0):,.0f}")
    col2.metric("Total Revenue", f"₹{insights.get('total_revenue', 0):,.0f}")
    col3.metric("ROI", f"{insights.get('roi', 0):.2f}")

    st.divider()

# ---------------- INSIGHT BUTTON ---------------- #
if uploaded_file and st.sidebar.button("Generate AI Insights"):
    with st.spinner("Generating insights..."):

        context_data = (
            st.session_state.df_insights
            if st.session_state.df_insights
            else file_content[:4000]
        )

        prompt = f"""
        You are a senior Marketing Analyst.

        DATA:
        {context_data}

        Provide:
        - Key insights
        - ROI evaluation
        - What is working / not working
        - 3 actionable recommendations
        """

        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
            ),
        )

        st.markdown("### 📊 AI Insights")
        st.write(response.text)

# ---------------- CHAT ---------------- #
st.subheader("💬 Ask Questions")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Ask about your data or general queries..."):
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):

        context_data = (
            st.session_state.df_insights
            if st.session_state.df_insights
            else file_content[:2000]
        )

        chat_prompt = f"""
        Context:
        {context_data}

        User Question:
        {user_input}
        """

        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=chat_prompt,
            config=types.GenerateContentConfig(
                system_instruction="You are a Marketing Analyst for data questions. Otherwise act like an IAS aspirant.",
                temperature=0.5,
            ),
        )

        reply = response.text
        st.markdown(reply)

        st.session_state.messages.append({"role": "assistant", "content": reply})
