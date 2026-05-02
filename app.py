import streamlit as st
import pandas as pd
from google import genai
from google.genai import types

# Optional PDF support
try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

# ---------------- CONFIG ---------------- #
st.set_page_config(page_title="Marketing Copilot AI", layout="wide")

# ---------------- API KEY ---------------- #
api_key = st.secrets.get("GEMINI_API_KEY") or st.sidebar.text_input("Enter API Key", type="password")

if not api_key:
    st.warning("Please provide an API Key")
    st.stop()

client = genai.Client(api_key=api_key)

# ---------------- SESSION ---------------- #
if "messages" not in st.session_state:
    st.session_state.messages = []

if "df" not in st.session_state:
    st.session_state.df = None

if "detected_cols" not in st.session_state:
    st.session_state.detected_cols = None

# ---------------- COLUMN DETECTION ---------------- #
COLUMN_PATTERNS = {
    "revenue": ["revenue", "sales", "gmv", "income"],
    "spend": ["spend", "cost", "investment"],
    "customers": ["customer", "users", "buyers"],
    "channel": ["channel", "source", "platform"],
    "date": ["date", "month", "time"]
}


def detect_columns(df):
    detected = {}
    for target, keywords in COLUMN_PATTERNS.items():
        for col in df.columns:
            col_clean = col.lower().strip()
            if any(k in col_clean for k in keywords):
                detected[target] = col
                break
    return detected


# ---------------- CLEANING ---------------- #
def clean_numeric(series):
    return pd.to_numeric(
        series.astype(str).str.replace(r"[^\d.-]", "", regex=True),
        errors="coerce"
    )


def clean_text(series):
    return series.astype(str).str.lower().str.strip()


def clean_dates(series):
    return pd.to_datetime(series, errors="coerce")


def auto_clean_pipeline(df):
    df.columns = [c.strip() for c in df.columns]
    detected_cols = detect_columns(df)
    clean_df = df.copy()

    for key, col in detected_cols.items():
        if key in ["revenue", "spend", "customers"]:
            clean_df[col] = clean_numeric(clean_df[col])
        elif key == "channel":
            clean_df[col] = clean_text(clean_df[col])
        elif key == "date":
            clean_df[col] = clean_dates(clean_df[col])

    return clean_df, detected_cols


# ---------------- KPI ---------------- #
def compute_kpis(df, cols):
    insights = {}

    rev = cols.get("revenue")
    spend = cols.get("spend")
    cust = cols.get("customers")

    if rev:
        insights["total_revenue"] = df[rev].sum()

    if spend:
        insights["total_spend"] = df[spend].sum()

    if rev and spend and insights["total_spend"] != 0:
        insights["roi"] = insights["total_revenue"] / insights["total_spend"]

    if cust:
        insights["total_customers"] = df[cust].sum()

    return insights


# ---------------- CHANNEL PERF ---------------- #
def channel_performance(df, cols):
    ch = cols.get("channel")
    rev = cols.get("revenue")
    spend = cols.get("spend")

    if not ch or not rev:
        return None

    agg = {rev: "sum"}
    if spend:
        agg[spend] = "sum"

    result = df.groupby(ch).agg(agg).reset_index()

    if spend:
        result["ROI"] = result[rev] / result[spend]

    return result.sort_values(by=rev, ascending=False)


# ---------------- SAFE CONTEXT ---------------- #
def safe_context(data, limit=3000):
    return str(data)[:limit] if data else "No data"


# ---------------- UI ---------------- #
st.title("📊 Marketing Copilot AI")

st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("CSV / PDF / TXT", type=["csv", "pdf", "txt"])

file_content = ""

if uploaded_file:
    st.sidebar.success("Uploaded")

    if uploaded_file.type == "text/csv":
        uploaded_file.seek(0)
        raw_df = pd.read_csv(uploaded_file)

        clean_df, detected_cols = auto_clean_pipeline(raw_df)

        st.session_state.df = clean_df
        st.session_state.detected_cols = detected_cols

    else:
        if uploaded_file.type == "application/pdf" and PdfReader:
            reader = PdfReader(uploaded_file)
            for page in reader.pages:
                file_content += page.extract_text() or ""
        else:
            file_content = uploaded_file.read().decode("utf-8")


# ---------------- DEBUG ---------------- #
if st.session_state.detected_cols:
    st.subheader("🔍 Detected Columns")
    st.json(st.session_state.detected_cols)


# ---------------- KPI ---------------- #
if st.session_state.df is not None:
    kpis = compute_kpis(st.session_state.df, st.session_state.detected_cols)

    st.subheader("📈 KPIs")
    c1, c2, c3 = st.columns(3)

    c1.metric("Revenue", f"₹{kpis.get('total_revenue', 0):,.0f}")
    c2.metric("Spend", f"₹{kpis.get('total_spend', 0):,.0f}")
    c3.metric("ROI", f"{kpis.get('roi', 0):.2f}")


# ---------------- CHANNEL ---------------- #
if st.session_state.df is not None:
    perf = channel_performance(st.session_state.df, st.session_state.detected_cols)

    if perf is not None:
        st.subheader("📊 Channel Performance")
        st.dataframe(perf, use_container_width=True)
        st.bar_chart(perf.set_index(perf.columns[0]))


# ---------------- AI INSIGHTS ---------------- #
if st.sidebar.button("Generate AI Insights"):
    context = safe_context(
        st.session_state.df if st.session_state.df is not None else file_content
    )

    prompt = f"""
    You are a Marketing Analyst.

    DATA:
    {context}

    Provide:
    - Key insights
    - Best channel
    - Worst channel
    - Budget suggestions
    """

    try:
        response = client.models.generate_content(
            model="gemini-1.5-flash-latest",
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.3),
        )
        st.write(response.text)

    except Exception as e:
        st.error(str(e))


# ---------------- CHAT ---------------- #
st.subheader("💬 Chat")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if q := st.chat_input("Ask..."):
    st.session_state.messages.append({"role": "user", "content": q})

    with st.chat_message("assistant"):

        context = safe_context(
            st.session_state.df if st.session_state.df is not None else file_content
        )

        try:
            res = client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=f"Context:\n{context}\n\nQuestion:{q}",
                config=types.GenerateContentConfig(temperature=0.5),
            )
            reply = res.text
        except Exception as e:
            reply = f"Error: {str(e)}"

        st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})
