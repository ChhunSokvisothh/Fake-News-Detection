import streamlit as st
import joblib
import re
import string

# ============================================================
# PAGE CONFIG - MUST BE FIRST!
# ============================================================
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

# ============================================================
# LOAD MODELS
# ============================================================
@st.cache_resource
def load_models():
    """Load all models and vectorizer (cached for performance)"""
    try:
        gb_model = joblib.load("model/gradient_boosting_model.pkl")
        rf_model = joblib.load("model/random_forest_model.pkl")
        dt_model = joblib.load("model/decision_tree_model.pkl")
        vectorizer = joblib.load("model/tfidf_vectorizer.pkl")
        return gb_model, rf_model, dt_model, vectorizer
    except FileNotFoundError as e:
        st.error(f"❌ Model files not found: {e}")
        st.stop()

gb_model, rf_model, dt_model, vectorizer = load_models()

# ============================================================
# TEXT PREPROCESSING (MUST MATCH TRAINING!)
# ============================================================
def wordopt(text):
    """Clean and normalize text - SAME as training script"""
    if not isinstance(text, str) or text.strip() == "":
        return ""
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub('\s+', ' ', text)
    return text.strip()

# ============================================================
# UI
# ============================================================
st.title("📰 Fake News Detection System")
st.caption("Final Year Project — Detect whether a news article is real or fake using AI")

st.markdown("---")

# Info box
with st.expander("ℹ️ How to use", expanded=False):
    st.markdown("""
    **You can enter:**
    - ✅ Just the article content (most common)
    - ✅ Title + content (best accuracy)
    - ✅ Just the title (quick check, less accurate)
    
    **Tips:**
    - Longer text = better accuracy
    - Include full articles when possible
    - Try different models to compare results
    """)

st.subheader("Enter News Details")

# Input fields
title = st.text_input(
    "📌 News Title (optional)",
    placeholder="e.g., Breaking: Major Event Happens",
    help="Optional, but helps improve accuracy"
)

content = st.text_area(
    "📝 News Content",
    height=250,
    placeholder="Paste the full article text here...",
    help="The main article text - required for best results"
)

# Model selection
col1, col2 = st.columns([2, 1])
with col1:
    model_choice = st.selectbox(
        "🤖 Select AI Model",
        ["Gradient Boosting", "Random Forest", "Decision Tree"],
        help="Gradient Boosting usually gives best results"
    )

with col2:
    st.metric("Features", f"{vectorizer.max_features:,}")

st.markdown("---")

# ============================================================
# PREDICTION
# ============================================================
if st.button("🔍 Analyze News Article", type="primary", use_container_width=True):
    # Validation
    if not title.strip() and not content.strip():
        st.warning("⚠️ Please enter either a title or content (or both) to analyze.")
    else:
        with st.spinner("🔄 Analyzing article..."):
            # Clean inputs
            title_clean = wordopt(title)
            content_clean = wordopt(content)
            
            # Combine title and content (same as training)
            # Title appears twice for emphasis (matches training script)
            combined_text = f"{title_clean} {title_clean} {content_clean}".strip()
            
            # Check if we have enough text
            if len(combined_text) < 10:
                st.error("❌ Text too short. Please provide more content.")
            else:
                # Vectorize
                text_vectorized = vectorizer.transform([combined_text])
                
                # Select model
                if model_choice == "Gradient Boosting":
                    model = gb_model
                elif model_choice == "Random Forest":
                    model = rf_model
                else:
                    model = dt_model
                
                # Predict
                prediction = model.predict(text_vectorized)[0]
                
                # Get confidence if available
                if hasattr(model, 'predict_proba'):
                    confidence = model.predict_proba(text_vectorized)[0]
                    fake_conf = confidence[0]
                    real_conf = confidence[1]
                else:
                    fake_conf = None
                    real_conf = None
                
                # Display results
                st.markdown("### 📊 Analysis Results")
                
                if prediction == 0:
                    st.error("### ❌ FAKE NEWS DETECTED")
                    st.markdown("This article shows characteristics of **fake or misleading news**.")
                else:
                    st.success("### ✅ APPEARS TO BE REAL NEWS")
                    st.markdown("This article shows characteristics of **legitimate news**.")
                
                # Show confidence if available
                if fake_conf is not None and real_conf is not None:
                    st.markdown("---")
                    st.markdown("#### Confidence Breakdown")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(
                            "Fake News",
                            f"{fake_conf:.1%}",
                            delta=None,
                            delta_color="inverse"
                        )
                    with col2:
                        st.metric(
                            "Real News",
                            f"{real_conf:.1%}",
                            delta=None
                        )
                    
                    # Progress bars
                    st.progress(fake_conf, text=f"Fake: {fake_conf:.1%}")
                    st.progress(real_conf, text=f"Real: {real_conf:.1%}")
                
                # Model info
                st.markdown("---")
                st.caption(f"🤖 Model used: **{model_choice}** | Features analyzed: **{text_vectorized.shape[1]:,}**")
                
                # What was analyzed
                with st.expander("🔍 What was analyzed?"):
                    if title.strip() and content.strip():
                        st.write("✅ Both title and content")
                    elif title.strip():
                        st.write("⚠️ Only title (consider adding content for better accuracy)")
                    else:
                        st.write("✅ Content only")
                    
                    st.write(f"**Total characters processed:** {len(combined_text):,}")
                    st.write(f"**Words analyzed:** {len(combined_text.split()):,}")

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🎓 <strong>Final Year Project</strong> — Fake News Detection System</p>
    <p>Built with Streamlit, scikit-learn, and TF-IDF vectorization</p>
    <p style='font-size: 0.8em; color: gray;'>
        💡 Tip: For best accuracy, use the Gradient Boosting model with full article text
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar with info
with st.sidebar:
    st.header("📚 About")
    st.markdown("""
    This AI system analyzes news articles to detect potential fake news using machine learning.
    
    **How it works:**
    1. Text preprocessing & cleaning
    2. TF-IDF vectorization
    3. ML model classification
    4. Confidence scoring
    
    **Models available:**
    - 🥇 Gradient Boosting (best)
    - 🥈 Random Forest (fast)
    - 🥉 Decision Tree (simple)
    """)
    
    st.markdown("---")
    
    st.header("⚙️ Model Info")
    st.write(f"**Features:** {vectorizer.max_features:,}")
    st.write(f"**Models loaded:** 3")
    st.write(f"**Vectorizer:** TF-IDF")
    
    st.markdown("---")
    st.caption("⚠️ This is an educational project. Always verify news from multiple trusted sources.")