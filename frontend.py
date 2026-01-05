
import streamlit as st
import requests
import base64
from io import BytesIO
from PIL import Image
import pandas as pd
import json
import datetime
import altair as alt

NEON_CSS = """
/* Smooth fade-in animation for all components */
* {
  animation: fadeIn 0.6s ease-in-out;
}

@keyframes fadeIn {
  0% { opacity: 0; transform: translateY(6px); }
  100% { opacity: 1; transform: translateY(0); }
}

/* Hover pulse effect for cards */
.neon-card:hover {
  transition: transform 0.25s ease, box-shadow 0.25s ease;
  transform: translateY(-4px) scale(1.01);
  box-shadow: 0 0 18px rgba(0,255,255,0.18), 0 6px 20px rgba(0,0,0,0.6);
}

/* Soft glow hover for neon buttons */
.stButton>button:hover {
  box-shadow: 0 0 12px rgba(0,255,255,0.5), 0 0 20px rgba(255,0,255,0.3);
}

:root{
  --bg: #0A0F1F;
  --card: linear-gradient(135deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
  --cyan: #00FFFF;
  --pink: #FF00FF;
  --lime: #A7FF00;
  --muted: #8b94a6;
}

body, .stApp {
  background: var(--bg);
  color: #e6f3ff;
}

/* Card */
.neon-card{
  padding: 18px;
  border-radius: 12px;
  background: var(--card);
  box-shadow: 0 6px 20px rgba(0,0,0,0.6), 0 0 18px rgba(0,255,255,0.03) inset;
  border: 1px solid rgba(255,255,255,0.04);
}

/* Neon headings */
.neon-h1{font-size:30px; color:var(--cyan); text-shadow:0 0 8px rgba(0,255,255,0.18);} 
.neon-h2{font-size:18px; color:var(--pink); text-shadow:0 0 6px rgba(255,0,255,0.12);} 

/* Buttons */
.stButton>button{
  border-radius: 10px;
  padding: 8px 14px;
  box-shadow: 0 6px 18px rgba(0,0,0,0.6);
  transition: all .12s ease-in-out;
}
.stButton>button:hover{transform: translateY(-2px);}

/* Neon accents by class */
.neon-accent-cyan{border:2px solid rgba(0,255,255,0.18); box-shadow:0 0 14px rgba(0,255,255,0.05);} 
.neon-accent-pink{border:2px solid rgba(255,0,255,0.12); box-shadow:0 0 14px rgba(255,0,255,0.03);} 
.neon-accent-lime{border:2px solid rgba(167,255,0,0.12); box-shadow:0 0 14px rgba(167,255,0,0.03);} 

.small-muted{color:var(--muted); font-size:12px}
"""


BACKEND_URL = "http://localhost:8000/evaluate"

def init_state():
    """Initialize session state variables"""
    if "last_evaluation" not in st.session_state:
        st.session_state["last_evaluation"] = None
    if "results_history" not in st.session_state:
        st.session_state["results_history"] = []
    if "page" not in st.session_state:
        st.session_state["page"] = "evaluation"


def decode_base64_image(b64str):
    try:
        b = base64.b64decode(b64str)
        return Image.open(BytesIO(b)).convert("RGBA")
    except Exception as e:
        st.error(f"Error decoding image: {e}")
        return None


def save_result(result):
    entry = result.copy()
    entry["timestamp"] = datetime.datetime.now().isoformat()
    st.session_state["results_history"].append(entry)
    st.success("✅ Result saved to history")


def display_evaluation_quick(data):
    if not data:
        st.info("No evaluation data to display.")
        return

    marks = data.get("marks", "N/A")
    breakdown = data.get("breakdown", {})
    errors = data.get("errors", [])
    annotated_b64 = data.get("annotated_image")
    cols = st.columns([1, 2])
    
    with cols[0]:
        st.markdown(
            f"<div class='neon-card neon-accent-cyan'>"
            f"<h3 style='color:var(--cyan);'>Total Marks</h3>"
            f"<h1 style='color:var(--cyan); text-shadow:0 0 12px rgba(0,255,255,0.2);'>{marks}</h1>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with cols[1]:
        if annotated_b64:
            img = decode_base64_image(annotated_b64)
            if img:
                st.image(img, caption="Annotated Image", use_container_width=True)
            else:
                st.info("Annotated image returned but couldn't decode it")
    if breakdown:
        st.write("### 📊 Component Breakdown")
        df = pd.DataFrame(list(breakdown.items()), columns=["Component", "Score"])
        st.table(df)

    if errors:
        st.write("### ⚠️ Detected Issues")
        for e in errors:
            st.markdown(
                f"<div class='neon-card neon-accent-lime' style='margin:5px 0;'>• {e}</div>", 
                unsafe_allow_html=True
            )

def page_flowchart_evaluation():
    st.markdown(
        f"<div class='neon-h1'>🎨 Hand-Drawn Flowchart Recognition</div>", 
        unsafe_allow_html=True
    )
    st.markdown(
        "<div class='small-muted'>Upload a hand-drawn flowchart and get automated evaluation marks.</div>", 
        unsafe_allow_html=True
    )
    st.write("")

    uploaded = st.file_uploader(
        "Upload flowchart image (jpg, png, pdf)", 
        type=["png", "jpg", "jpeg", "pdf"]
    )
    if uploaded is not None:
        try:
            img = Image.open(uploaded)
            st.image(img, caption="Uploaded Image", use_container_width=True)
        except Exception:
            st.warning("⚠️ Couldn't preview the uploaded file")

    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("🔍 Evaluate Flowchart", type="primary"):
            if uploaded is None:
                st.warning("⚠️ Please upload an image first")
            else:
                with st.spinner("🔄 Analyzing flowchart..."):
                    try:
                        files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
                        
                        resp = requests.post(BACKEND_URL, files=files, timeout=60)
                        
                        if resp.status_code == 200:
                            data = resp.json()
                            st.session_state["last_evaluation"] = data
                            st.success(f"✅ Evaluation complete: {data.get('marks', 'N/A')} marks")
                            st.markdown("---")
                            display_evaluation_quick(data)
                            
                        else:
                            st.error(f"❌ Backend error: {resp.status_code}")
                            st.error(f"Details: {resp.text}")
                            
                    except requests.exceptions.ConnectionError:
                        st.error("❌ Cannot connect to backend. Make sure the API is running on http://localhost:8000")
                        st.info("Run: python backend.py")
                        
                    except requests.exceptions.Timeout:
                        st.error("❌ Request timed out. The image may be too large or complex.")
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")

    with col2:
        if st.button("💾 Save Result"):
            if st.session_state.get("last_evaluation"):
                save_result(st.session_state["last_evaluation"])
            else:
                st.warning("⚠️ No evaluation to save")
    if st.session_state.get("last_evaluation"):
        st.markdown("---")
        st.header("📋 Last Evaluation")
        display_evaluation_quick(st.session_state["last_evaluation"])
        
        if st.button("📊 View Detailed Breakdown"):
            st.session_state["page"] = "breakdown"
            st.rerun()
def page_breakdown():
    st.markdown(
        f"<div class='neon-h2'>📊 Evaluation Breakdown</div>", 
        unsafe_allow_html=True
    )
    
    data = st.session_state.get("last_evaluation")
    
    if not data:
        st.warning("⚠️ No evaluation found. Run an evaluation first.")
        if st.button("← Back to Evaluation"):
            st.session_state["page"] = "evaluation"
            st.rerun()
        return

    breakdown = data.get("breakdown", {})
    df = pd.DataFrame(list(breakdown.items()), columns=["component", "marks"])
    st.markdown("<div class='neon-card neon-accent-pink'>", unsafe_allow_html=True)
    st.table(df)
    st.markdown("</div>", unsafe_allow_html=True)
    if not df.empty:
        chart = alt.Chart(df).mark_bar(color='#00FFFF').encode(
            x=alt.X('component:N', sort=None, title='Component'),
            y=alt.Y('marks:Q', title='Marks')
        ).properties(
            height=400
        )
        st.altair_chart(chart, use_container_width=True)
    st.write("### ⚠️ Detected Errors")
    errors = data.get("errors", [])
    if errors:
        for e in errors:
            st.markdown(
                f"<div class='neon-card neon-accent-lime'>• {e}</div>", 
                unsafe_allow_html=True
            )
    else:
        st.success("✅ No errors detected!")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back to Evaluation"):
            st.session_state["page"] = "evaluation"
            st.rerun()
    with col2:
        if st.button("📖 View Symbol Guide"):
            st.session_state["page"] = "symbols"
            st.rerun()
def page_symbols():
    st.markdown(
        f"<div class='neon-h1' style='margin-bottom:12px;'>📖 Flowchart Symbol Guide</div>", 
        unsafe_allow_html=True
    )
    st.write("Reference for common flowchart symbols and correct usage")

    st.markdown(
        "<div class='neon-card neon-accent-cyan'>"
        "<b>⭕ Start / Stop (Oval)</b><br/>"
        "Represents entry/exit point of the flowchart."
        "</div>", 
        unsafe_allow_html=True
    )
    
    st.markdown(
        "<div class='neon-card neon-accent-pink'>"
        "<b>▱ Input / Output (Parallelogram)</b><br/>"
        "Use for reading input or displaying output."
        "</div>", 
        unsafe_allow_html=True
    )
    
    st.markdown(
        "<div class='neon-card neon-accent-lime'>"
        "<b>▭ Process (Rectangle)</b><br/>"
        "Represents an action, operation, or process step."
        "</div>", 
        unsafe_allow_html=True
    )
    
    st.markdown(
        "<div class='neon-card' style='border-color:rgba(0,120,255,0.12)'>"
        "<b>◆ Decision (Diamond)</b><br/>"
        "Branching condition with yes/no or true/false flows."
        "</div>", 
        unsafe_allow_html=True
    )

    st.write("### ✅ Best Practices")
    st.markdown("""
    - **Clear Labels**: All shapes should have descriptive text
    - **Single Entry/Exit**: Use one start and one stop symbol
    - **Proper Arrows**: Arrows should be clear and unambiguous
    - **Decision Labels**: Label decision branches (Yes/No, True/False)
    - **Consistent Flow**: Top-to-bottom or left-to-right flow
    """)

    st.write("### ❌ Common Mistakes")
    st.markdown("""
    - Unlabeled shapes or unclear text
    - Missing arrows between connected steps
    - Crossed or ambiguous arrow paths
    - Multiple start/stop points
    - Decision diamonds without branch labels
    """)

    if st.button("← Return to Evaluation"):
        st.session_state["page"] = "evaluation"
        st.rerun()

def page_history():
    """Results history page"""
    st.markdown(
        f"<div class='neon-h2'>🗂️ Results History</div>", 
        unsafe_allow_html=True
    )
    
    history = st.session_state.get("results_history", [])
    
    if not history:
        st.markdown(
            "<div class='neon-card neon-accent-pink'>⚠️ No evaluations saved yet!</div>", 
            unsafe_allow_html=True
        )
        return

    display_rows = []
    for i, e in enumerate(history):
        display_rows.append({
            "Index": i,
            "Timestamp": e.get("timestamp", "-"),
            "Marks": e.get("marks", "-")
        })
    
    df = pd.DataFrame(display_rows)
    st.dataframe(df, use_container_width=True)
    st.write("### 🔧 Manage Results")
    sel = st.number_input(
        "Select index to view/manage", 
        min_value=0, 
        max_value=len(history)-1, 
        value=0
    )
    
    entry = history[int(sel)]
    
    st.write("**Selected Entry:**")
    st.json(entry)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🗑️ Delete Selected"):
            history.pop(int(sel))
            st.success("✅ Entry deleted")
            st.rerun()
    
    with col2:
        json_str = json.dumps(entry, indent=2)
        st.download_button(
            "📥 Download JSON",
            json_str,
            file_name=f"evaluation_{int(sel)}.json",
            mime="application/json"
        )
    
    with col3:
        if st.button("📋 Load as Current"):
            st.session_state["last_evaluation"] = entry
            st.success("✅ Loaded into Last Evaluation")
            st.session_state["page"] = "evaluation"
            st.rerun()
def main():
    st.set_page_config(
        page_title="Flowchart Recognition System",
        page_icon="🎨",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown(f"<style>{NEON_CSS}</style>", unsafe_allow_html=True)
    
    init_state()
    st.sidebar.markdown(
        "<h2 style='color:var(--cyan); margin-bottom:12px;'>🧭 Navigation</h2>", 
        unsafe_allow_html=True
    )

    nav_items = [
        ("evaluation", "🏠 Evaluation"),
        ("breakdown", "📊 Breakdown"),
        ("symbols", "📖 Symbol Guide"),
        ("history", "🗂️ Results History")
    ]

    for nav_key, nav_label in nav_items:
        if st.sidebar.button(nav_label, use_container_width=True):
            st.session_state["page"] = nav_key
            st.rerun()
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "<div class='small-muted'>"
        "💡 <b>Tip:</b> Make sure your backend is running on port 8000"
        "</div>",
        unsafe_allow_html=True
    )
    page = st.session_state.get("page", "evaluation")
    st.markdown("<div style='animation:fadeIn 0.45s ease;'>", unsafe_allow_html=True)

    if page == "evaluation":
        page_flowchart_evaluation()
    elif page == "breakdown":
        page_breakdown()
    elif page == "symbols":
        page_symbols()
    elif page == "history":
        page_history()

    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()