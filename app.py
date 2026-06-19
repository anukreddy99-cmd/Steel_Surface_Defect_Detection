import streamlit as st
import cv2
import numpy as np
import joblib
from PIL import Image
from skimage.feature import hog
import os  

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Steel Surface Defect Detection",
    page_icon="🔍",
    layout="wide" 
)

# ---------------- Modern CSS ----------------
st.markdown("""
<style>

.stApp{
background: linear-gradient(135deg,#0F2027,#203A43,#2C5364);
}

.main-title{
text-align:center;
font-size:55px;
font-weight:bold;
color:white;
}

.sub-title{
text-align:center;
font-size:20px;
color:#EAEAEA;
}

.block-container{
padding-top:1rem;
}

[data-testid="stSidebar"]{
background: rgba(255,255,255,0.08);
backdrop-filter: blur(15px);
}

[data-testid="stMetric"]{
background: rgba(255,255,255,0.08);
padding:15px;
border-radius:20px;
box-shadow:0px 5px 20px rgba(0,0,0,0.3);
}

h1,h2,h3{
color:white;
}

p{
color:white;
}

div[data-testid="stMarkdownContainer"]{
color:white;
}

.stSuccess{
border-radius:15px;
}

</style>
""", unsafe_allow_html=True)

st.markdown(
"""
<div class="main-title">
🔍 Steel Surface Defect Detection
</div>

<div class="sub-title">
AI-powered Quality Inspection System
</div>
""",
unsafe_allow_html=True
)

# ---------------- Sidebar ----------------

st.sidebar.image(
"https://cdn-icons-png.flaticon.com/512/4139/4139981.png",
width=120
)

st.sidebar.title("About Project")

st.sidebar.markdown("""
### 🎯 Defects Detected

✅ Crazing

✅ Inclusion

✅ Patches

✅ Pitted Surface

✅ Rolled-in Scale

✅ Scratches

---

### 🤖 Model

Gaussian Naive Bayes

---

### 🔬 Feature Extraction

HOG Features

---

### 👩‍💻 Developed By

Hansika
""")

# ---------------- Load Model ----------------

@st.cache_resource
def load_model():
    return joblib.load("surface_defect_model.pkl")

model = load_model()

# ---------------- Classes ----------------

class_names = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches"
]

# ---------------- Reference Images ----------------

references = {
    "crazing": "references/crazing_img.jpeg",
    "inclusion": "references/inclusion_img.jpeg",
    "patches": "references/patches_img.jpeg",
    "pitted_surface": "references/pitted_surface_img.jpeg",
    "rolled-in_scale": "references/rolled_in_scale_img.jpeg",
    "scratches": "references/scratch_img.jpeg"
}

# ---------------- Descriptions ----------------

descriptions = {
    "crazing":"Fine crack-like lines on the steel surface.",
    "inclusion":"Foreign particles trapped inside steel.",
    "patches":"Uneven regions or discoloration on the surface.",
    "pitted_surface":"Small holes or pits present on steel.",
    "rolled-in_scale":"Oxide scale pressed into steel during rolling.",
    "scratches":"Visible lines caused by friction."
}

impacts = {
    "crazing":"Can reduce durability and lead to failure.",
    "inclusion":"May affect strength and reliability.",
    "patches":"Poor surface finish and appearance.",
    "pitted_surface":"Can initiate corrosion.",
    "rolled-in_scale":"May cause coating defects.",
    "scratches":"Reduces visual quality."
}

recommendations = {
    "crazing":"Inspect and reject severe defects.",
    "inclusion":"Improve raw material quality.",
    "patches":"Surface treatment recommended.",
    "pitted_surface":"Polish or reject severe cases.",
    "rolled-in_scale":"Improve rolling process.",
    "scratches":"Polish surface if possible."
}

# ---------------- Upload ----------------

uploaded_file = st.file_uploader(
    "Upload Steel Surface Image",
    type=["jpg","jpeg","png"]
) 

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    img = np.array(image)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    gray = cv2.resize(gray, (128,128))

    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8,8),
        cells_per_block=(2,2)
    )

    features = features.reshape(1,-1)

    prediction = model.predict(features)

    predicted_class = class_names[int(prediction[0])]

    st.markdown("## 🎯 Prediction Result")

    st.success(
        f"✅ Defect Detected : {predicted_class.upper()}"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("""
        <div style="
        background:rgba(255,255,255,0.08);
        padding:15px;
        border-radius:20px;
        text-align:center;
        ">
        <h3>📤 Uploaded Image</h3>
        </div>
        """,
        unsafe_allow_html=True)

        st.image(image, use_container_width=True)

    with col2:

        st.markdown("""
        <div style="
        background:rgba(255,255,255,0.08);
        padding:15px;
        border-radius:20px;
        text-align:center;
        ">
        <h3>📌 Typical Reference Image</h3>
        </div>
        """,
        unsafe_allow_html=True)

        if os.path.exists(references[predicted_class]):
            st.write("Prediction:", predicted_class)
            st.write("Path:", references[predicted_class]) 
            st.image(
                references[predicted_class],
                caption=predicted_class.upper(),
                use_container_width=True
            )
        else:
            st.warning("Reference image not found.")

    st.markdown("---")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "🔍 Defect",
            predicted_class.upper()
        )

    with c2:
        st.metric(
            "🤖 Model",
            "GaussianNB"
        )

    with c3:
        st.metric(
            "🧠 Feature",
            "HOG"
        )

    st.markdown("---")

        # ---------------- Premium Information Cards ----------------

    st.markdown("## 📋 Defect Analysis")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(f"""
        <div style="
        background:rgba(255,255,255,0.10);
        padding:25px;
        border-radius:20px;
        box-shadow:0px 5px 20px rgba(0,0,0,0.3);
        height:250px;
        ">
        <h3>🔍 Description</h3>
        <p>{descriptions[predicted_class]}</p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div style="
        background:rgba(255,255,255,0.10);
        padding:25px;
        border-radius:20px;
        box-shadow:0px 5px 20px rgba(0,0,0,0.3);
        height:250px;
        ">
        <h3>⚠ Quality Impact</h3>
        <p>{impacts[predicted_class]}</p>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div style="
        background:rgba(255,255,255,0.10);
        padding:25px;
        border-radius:20px;
        box-shadow:0px 5px 20px rgba(0,0,0,0.3);
        height:250px;
        ">
        <h3>💡 Recommendation</h3>
        <p>{recommendations[predicted_class]}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <hr style="border:1px solid rgba(255,255,255,0.2);">

    <div style='text-align:center; color:white;'>

    ### 🚀 AI-Based Steel Surface Defect Detection System

    Developed by <b>Hansika</b> ❤️

    Gaussian Naive Bayes + HOG Features | Streamlit Dashboard

    </div>
    """, unsafe_allow_html=True)

    st.balloons()
