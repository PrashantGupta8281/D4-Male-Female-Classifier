import streamlit as st
import numpy as np
from PIL import Image
import joblib

# --------------------------------
# Page Config
# --------------------------------
st.set_page_config(
    page_title="Male vs Female Classifier",
    page_icon="🧑",
    layout="centered"
)

# --------------------------------
# Custom CSS
# --------------------------------
st.markdown("""
<style>

.stApp{
    background: linear-gradient(135deg,#6dd5ed,#2193b0);
}

.main-title{
    font-size:42px;
    font-weight:bold;
    text-align:center;
    color:white;
}

.subtitle{
    text-align:center;
    font-size:18px;
    color:white;
    margin-bottom:25px;
}

.card{
    background:rgba(255,255,255,0.18);
    padding:30px;
    border-radius:20px;
    backdrop-filter: blur(12px);
    box-shadow:0px 10px 25px rgba(0,0,0,0.25);
}

.result{
    text-align:center;
    font-size:28px;
    font-weight:bold;
    color:#ffffff;
}

.css-1v0mbdj img{
    border-radius:20px;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------
# Sidebar
# --------------------------------
st.sidebar.title("📌 About")
st.sidebar.info(
"""
This AI model predicts whether an uploaded face belongs to:

👨 Male

👩 Female

Built using:
- Logistic Regression
- Scikit-Learn
- Streamlit
"""
)

# --------------------------------
# Load Model
# --------------------------------
model = joblib.load("Male_And_Female_model.pkl")

IMG_SIZE = 64

# --------------------------------
# Header
# --------------------------------
st.markdown(
'<p class="main-title">👨 Male vs 👩 Female Classifier</p>',
unsafe_allow_html=True
)

st.markdown(
'<p class="subtitle">Upload an image and let AI classify it instantly.</p>',
unsafe_allow_html=True
)

st.markdown('<div class="card">', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "📷 Upload Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1,1])

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    resized = image.resize((IMG_SIZE, IMG_SIZE))
    resized = np.array(resized).flatten()

    prediction = model.predict([resized])[0]
    probability = model.predict_proba([resized])[0]

    with col2:

        st.subheader("Prediction")

        if prediction == 0:
            st.success("👨 Male")
        else:
            st.success("👩 Female")

        st.balloons()

        st.markdown("### Confidence")

        st.write("👨 Male")
        st.progress(float(probability[0]))

        st.write(f"**{probability[0]*100:.2f}%**")

        st.write("👩 Female")
        st.progress(float(probability[1]))

        st.write(f"**{probability[1]*100:.2f}%**")

    st.divider()

    st.metric(
        label="Highest Confidence",
        value=f"{max(probability)*100:.2f}%"
    )

st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
"""
<br><br>
<center>
Made with ❤️ using Streamlit
</center>
""",
unsafe_allow_html=True
)
