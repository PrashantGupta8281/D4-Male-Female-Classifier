import streamlit as st
import numpy as np
from PIL import Image
import joblib
import plotly.graph_objects as go

# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(
    page_title="AI Gender Classification",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------
# Custom CSS
# -------------------------
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap" rel="stylesheet">

<style>

html, body, [class*="css"]{
    font-family:'Outfit', sans-serif;
}

.stApp{
background:
radial-gradient(circle at top left,#3b82f6 0%,transparent 35%),
radial-gradient(circle at bottom right,#7c3aed 0%,transparent 35%),
linear-gradient(135deg,#020617,#0f172a);
color:white;
}

.block-container{
padding-top:2rem;
padding-bottom:2rem;
}

.main-title{
font-size:60px;
font-weight:700;
text-align:center;
background:linear-gradient(90deg,#38BDF8,#A855F7);
-webkit-background-clip:text;
-webkit-text-fill-color:transparent;
}

.subtitle{
text-align:center;
font-size:20px;
color:#CBD5E1;
margin-bottom:30px;
}

.glass{
background:rgba(255,255,255,.08);
backdrop-filter:blur(18px);
padding:30px;
border-radius:25px;
border:1px solid rgba(255,255,255,.15);
box-shadow:0 8px 40px rgba(0,0,0,.4);
}

div[data-testid="stImage"] img{
border-radius:18px;
border:3px solid #38BDF8;
box-shadow:0 0 25px rgba(56,189,248,.45);
}

.stButton>button{
background:linear-gradient(90deg,#2563EB,#7C3AED);
color:white;
font-weight:600;
border-radius:40px;
padding:.6rem 2rem;
border:none;
}

.stProgress>div>div{
background:linear-gradient(90deg,#38BDF8,#A855F7);
}

</style>
""", unsafe_allow_html=True)

# -------------------------
# Sidebar
# -------------------------
st.sidebar.title("🧠 AI Gender Classification")

st.sidebar.markdown("""
### Model

- Logistic Regression
- Image Size: 64×64
- RGB Images
- Binary Classification

---

### Features

✅ Upload Images

✅ Real-Time Prediction

✅ Confidence Scores

✅ Interactive Dashboard

---

Built with ❤️ using Streamlit
""")

# -------------------------
# Load Model
# -------------------------
model = joblib.load("Male_And_Female_model.pkl")

IMG_SIZE = 64

# -------------------------
# Header
# -------------------------
st.markdown(
'<div class="main-title">AI Gender Classification</div>',
unsafe_allow_html=True
)

st.markdown(
'<div class="subtitle">Upload an image and let Artificial Intelligence classify it instantly.</div>',
unsafe_allow_html=True
)

st.markdown('<div class="glass">', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "📷 Upload Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1,1])

    with col1:
        st.image(image, use_container_width=True)

    resized = image.resize((IMG_SIZE, IMG_SIZE))
    resized = np.array(resized).flatten()

    prediction = model.predict([resized])[0]
    probability = model.predict_proba([resized])[0]

    with col2:

        st.subheader("Prediction")

        if prediction == 0:

            st.markdown("""
            <div style="
            background:linear-gradient(90deg,#2563EB,#06B6D4);
            padding:20px;
            border-radius:15px;
            text-align:center;
            font-size:30px;
            font-weight:700;">
            👨 Male
            </div>
            """, unsafe_allow_html=True)

        else:

            st.markdown("""
            <div style="
            background:linear-gradient(90deg,#A855F7,#EC4899);
            padding:20px;
            border-radius:15px;
            text-align:center;
            font-size:30px;
            font-weight:700;">
            👩 Female
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### Confidence")

        st.write(f"👨 Male: **{probability[0]*100:.2f}%**")
        st.progress(float(probability[0]))

        st.write(f"👩 Female: **{probability[1]*100:.2f}%**")
        st.progress(float(probability[1]))

        st.metric(
            "Highest Confidence",
            f"{max(probability)*100:.2f}%"
        )

    st.markdown("---")

    fig = go.Figure(data=[go.Pie(
        labels=["Male","Female"],
        values=probability,
        hole=.70,
        marker=dict(colors=["#38BDF8","#A855F7"])
    )])

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", family="Outfit"),
        height=420
    )

    st.plotly_chart(fig, use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
"""
<hr>

<div style="text-align:center;color:#94A3B8">
AI Gender Classification • Competition Project • Built with Streamlit
</div>
""",
unsafe_allow_html=True
)
