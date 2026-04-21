import json
import os

import google.generativeai as genai
import PyPDF2
import streamlit as st

st.set_page_config(page_title="Resume Analyzer", page_icon="📄", layout="centered")

def _get_api_key() -> str | None:
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GEMINI_API_KEY")


API_KEY = _get_api_key()
if not API_KEY:
    st.error("GEMINI_API_KEY is not set. Add it to your Streamlit secrets or environment variables.")
    st.stop()

genai.configure(api_key=API_KEY)


def extract_pdf_text(uploaded_file) -> str:
    reader = PyPDF2.PdfReader(uploaded_file)
    text_parts = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        text_parts.append(page_text)
    return "\n".join(text_parts).strip()


def analyze_resume(resume_text: str, job_role: str | None) -> dict:
    role_line = f"The candidate is targeting the role: {job_role}." if job_role else "No specific target role was provided; evaluate as a general professional resume."

    prompt = f"""You are an expert technical recruiter and career coach.
Analyze the following resume and return ONLY a JSON object (no prose, no markdown fences) with this exact schema:

{{
  "score": number,                       // overall quality on a 0-10 scale, one decimal allowed
  "summary": "string",                   // 2-3 sentence overall summary
  "strengths": ["string", ...],          // 3-6 concrete strengths
  "weaknesses": ["string", ...],         // 3-6 concrete weaknesses or gaps
  "improvements": ["string", ...]        // 4-7 actionable improvement suggestions
}}

{role_line}

Resume content:
\"\"\"
{resume_text}
\"\"\"
"""

    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(
        prompt,
        generation_config={"response_mime_type": "application/json"},
    )

    raw = response.text.strip()
    return json.loads(raw)


def render_score(score: float) -> None:
    score = max(0.0, min(10.0, float(score)))
    if score >= 8:
        color = "#16a34a"
        label = "Excellent"
    elif score >= 6:
        color = "#ca8a04"
        label = "Good"
    elif score >= 4:
        color = "#ea580c"
        label = "Needs Work"
    else:
        color = "#dc2626"
        label = "Weak"

    st.markdown(
        f"""
        <div style="text-align:center;padding:24px;border-radius:12px;border:1px solid #e5e7eb;">
            <div style="font-size:14px;color:#6b7280;text-transform:uppercase;letter-spacing:1px;">Resume Score</div>
            <div style="font-size:64px;font-weight:700;color:{color};line-height:1;">{score:.1f}<span style="font-size:24px;color:#9ca3af;">/10</span></div>
            <div style="font-size:16px;color:{color};font-weight:600;margin-top:4px;">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(score / 10.0)


st.title("📄 Resume Analyzer")
st.caption("Upload your resume and get an AI-powered evaluation powered by Google Gemini.")

with st.sidebar:
    st.header("About")
    st.write(
        "This tool reviews your resume and gives you a score out of 10 along with strengths, "
        "weaknesses, and concrete suggestions for improvement."
    )
    st.write("Your resume is sent to Google Gemini for analysis and is not stored.")

uploaded_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])
job_role = st.text_input("Target role (optional)", placeholder="e.g. Senior Backend Engineer")

analyze_clicked = st.button("Analyze Resume", type="primary", disabled=uploaded_file is None)

if analyze_clicked and uploaded_file is not None:
    with st.spinner("Reading your resume..."):
        try:
            resume_text = extract_pdf_text(uploaded_file)
        except Exception as exc:
            st.error(f"Could not read the PDF: {exc}")
            st.stop()

    if not resume_text or len(resume_text) < 50:
        st.error("The PDF appears to be empty or unreadable. Make sure it contains selectable text (not just scanned images).")
        st.stop()

    with st.spinner("Analyzing with Gemini..."):
        try:
            result = analyze_resume(resume_text, job_role.strip() or None)
        except json.JSONDecodeError:
            st.error("The AI response could not be parsed. Please try again.")
            st.stop()
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            st.stop()

    st.session_state["last_result"] = result
    st.session_state["last_resume_chars"] = len(resume_text)

if "last_result" in st.session_state:
    result = st.session_state["last_result"]

    st.divider()
    render_score(result.get("score", 0))

    summary = result.get("summary")
    if summary:
        st.subheader("Overall Summary")
        st.write(summary)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("✅ Strengths")
        for item in result.get("strengths", []):
            st.markdown(f"- {item}")

    with col2:
        st.subheader("⚠️ Weaknesses")
        for item in result.get("weaknesses", []):
            st.markdown(f"- {item}")

    st.subheader("💡 Suggestions for Improvement")
    for item in result.get("improvements", []):
        st.markdown(f"- {item}")

    with st.expander("View raw analysis (JSON)"):
        st.json(result)
