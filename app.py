import streamlit as st
import pandas as pd
from google import genai
from google.genai import types

# Optional PDF support (safe import)
try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

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

# ---------------- SESSION ---------------- #
if "messages" not in st.session_state:
    st.session_state.messages = []

if "df_insights" not in st.session_state:
    st.session_state.df_insights = None

if "channel_df" not in st.session_state:
    st.session_state.channel_df = None

# ---------------- UTILS ---------------- #
def safe_context(data, limit=3000):
    return str(data)[:limit] if data else "No data available"


def extract_file_content(uploaded_file):
    if uploaded_file.type == "application/pdf" and PdfReader:
        reader = PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    else:
        uploaded_file.seek(0)
        return uploaded_file.read().decode("utf-8")


def detect_column(df, keywords):
    for col in df.columns:
        for k in keywords:
            if k in col:
                return col
    return None


def process_campaign_data(df):
    df.columns = [c.strip().lower() for c in df.columns]

    insights = {}

    spend_col = detect_column(df, ["spend", "cost"])
    revenue_col = detect_column(df, ["revenue", "sales"])
    customer_col = detect_column(df, ["customer"])

    if spend_col:
        insights["total_spend"] = float(df[spend_col].sum())
    if revenue_col:
        insights["total_revenue"] = float(df[revenue_col].sum())

    if spend_col and revenue_col and insights["total_spend"] != 0:
        insights["roi"] = insights["total_revenue"] / insights["total_spend"]

    if customer_col:
        insights["total_customers"] = int(df[customer_col].sum())

    return insights


def channel_performance(df):
    df.columns = [c.strip().lower() for c in df.columns]

    channel_col = detect_column(df, ["channel"])
    revenue_col = detect_column(df, ["revenue", "sales"])
    spend_col = detect_column(df, ["spend", "cost"])

    if not channel_col or not revenue_col:
        return None

    agg_dict = {revenue_col: "sum"}
    if spend_col:
        agg_dict[spend_col] = "sum"

    result = df.groupby(channel_col).agg(agg_dict).reset_index()

    if spend_col:
        result["ROI"] = result[revenue_col] / result[spend_col]

    return result.sort_values(by=revenue_col, ascending=False)


# ---------------- SIDEBAR ---------------- #
st.sidebar.header("📁 Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV / PDF / TXT", type=["csv", "pdf", "txt"])

file_content = ""
df = None

if uploaded_file:
    st.sidebar.success("File uploaded!")

    if uploaded_file.type == "text/csv":
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file)

        # Process KPIs
        st.session_state.df_insights = process_campaign_data(df)

        # Channel performance
        st.session_state.channel_df = channel_performance(df)

    else:
        file_content = extract_file_content(uploaded_file)

# ---------------- KPI DISPLAY ---------------- #
if st.session_state.df_insights:
    insights = st.session_state.df_insights

    st.subheader("📈 Key Metrics")
    col1, col2, col3 = st.columns(3)

    col1.metric("Total Spend", f"₹{insights.get('total_spend', 0):,.0f}")
    col2.metric("Total Revenue", f"₹{insights.get('total_revenue', 0):,.0f}")
    col3.metric("ROI", f"{insights.get('roi', 0):.2f}")

    st.divider()

# ---------------- CHANNEL PERFORMANCE ---------------- #
if st.session_state.channel_df is not None:
    st.subheader("📊 Channel Performance")

    perf_df = st.session_state.channel_df

    st.dataframe(perf_df, use_container_width=True)

    st.bar_chart(perf_df.set_index(perf_df.columns[0]))

    st.divider()

# ---------------- AI INSIGHTS ---------------- #
if st.sidebar.button("Generate AI Insights"):
    with st.spinner("Analyzing..."):

        context_data = safe_context(
            st.session_state.channel_df if st.session_state.channel_df is not None
            else st.session_state.df_insights if st.session_state.df_insights
            else file_content
        )

        prompt = f"""
        You are a senior Marketing Analyst.

        DATA:
        {context_data}

        Provide:
        - Best performing channel
        - Worst performing channel
        - ROI insights
        - Budget reallocation suggestion
        """

        try:
            response = client.models.generate_content(
                model="gemini-1.5-flash-latest",
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3,
                ),
            )
            st.markdown("### 📊 AI Insights")
            st.write(response.text)

        except Exception as e:
            st.error(f"Error: {str(e)}")

# ---------------- CHAT ---------------- #
st.subheader("💬 Chat")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Ask about your data..."):
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):

        context_data = safe_context(
            st.session_state.channel_df if st.session_state.channel_df is not None
            else st.session_state.df_insights if st.session_state.df_insights
            else file_content
        )

        chat_prompt = f"""
        Context:
        {context_data}

        Question:
        {user_input}
        """

        try:
            response = client.models.generate_content(
                model="gemini-1.5-flash-latest",
                contents=chat_prompt,
                config=types.GenerateContentConfig(
                    system_instruction="You are a Marketing Analyst.",
                    temperature=0.5,
                ),
            )

            reply = response.text

        except Exception as e:
            reply = f"⚠️ Error: {str(e)}"

        st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})
