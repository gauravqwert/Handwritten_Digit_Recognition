import streamlit as st
from streamlit_drawable_canvas import st_canvas
import joblib
import numpy as np
from PIL import Image, ImageOps
import plotly.express as px

# ========== CUSTOM STYLING ==========
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

:root {
    --primary: #667eea;
    --secondary: #764ba2;
}

/* Main app styling */
.stApp {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    font-family: 'Poppins', sans-serif;
}

/* Canvas styling */
canvas {
    border: 2px solid var(--primary) !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1) !important;
    background-color: #000000 !important;
}

/* Prediction card */
.prediction-card {
    background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
    color: white;
    border-radius: 15px;
    padding: 20px;
    margin: 20px 0;
    box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.02); }
    100% { transform: scale(1); }
}

/* Tabs styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
}

.stTabs [data-baseweb="tab"] {
    background: white;
    border-radius: 10px 10px 0 0 !important;
    padding: 10px 20px !important;
    transition: all 0.3s;
}

.stTabs [aria-selected="true"] {
    background: var(--primary) !important;
    color: white !important;
}

/* Upload box */
.upload-box {
    background: white;
    border-radius: 15px;
    padding: 30px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    margin: 20px 0;
}
</style>
""", unsafe_allow_html=True)

# ========== MODEL LOADING ==========
@st.cache_resource
def load_model():
    return joblib.load('knn_digit_classifier.pkl')

model = load_model()

# ========== APP LAYOUT ==========
st.title("✍️ Interactive Digit Recognizer")

tab1, tab2 = st.tabs(["📁 Upload Image", "✏️ Draw Digit"])

def predict_digit(image):
    """Process image and return prediction"""
    img = ImageOps.grayscale(image)
    img = ImageOps.invert(img)  # MNIST uses white digits on black bg
    img = img.resize((8, 8))
    img_array = np.array(img).reshape(1, -1)
    prediction = model.predict(img_array)[0]
    confidence = np.max(model.predict_proba(img_array)) * 100
    probs = model.predict_proba(img_array)[0]
    return img, prediction, confidence, probs

# ========== UPLOAD TAB ==========
with tab1:
    st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a handwritten digit (0-9):", 
                                   type=["png", "jpg", "jpeg"])
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_file:
        img = Image.open(uploaded_file)
        processed_img, prediction, confidence, probs = predict_digit(img)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(processed_img, caption="Processed Image", width=150)
        with col2:
            st.markdown(f"""
            <div class='prediction-card'>
                <h2>Predicted Digit: {prediction}</h2>
                <p>Confidence: {confidence:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Probability chart
        fig = px.bar(
            x=range(10),
            y=probs,
            labels={'x': 'Digit', 'y': 'Probability'},
            color=probs,
            color_continuous_scale='thermal',
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)

# ========== DRAWING TAB ==========
with tab2:
    st.markdown("### Draw a digit (0-9)")
    
    # Create a key for the canvas that changes when we want to clear it
    if 'canvas_key' not in st.session_state:
        st.session_state.canvas_key = 0

    # Canvas with proper settings
    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",  # Transparent
        stroke_width=20,
        stroke_color="#FFFFFF",  # White drawing
        background_color="#000000",  # Black canvas
        height=200,
        width=200,
        drawing_mode="freedraw",
        key=f"canvas_{st.session_state.canvas_key}",
        update_streamlit=True
    )
    
    if canvas_result.image_data is not None:
        # Fixed version without deprecation warning
        img = Image.fromarray(canvas_result.image_data.astype('uint8')[..., :3])  # Remove alpha channel if present
        img = img.convert('RGBA')  # Explicit conversion to RGBA
        
        processed_img, prediction, confidence, probs = predict_digit(img)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(processed_img, caption="Your Drawing", width=150)
        with col2:
            st.markdown(f"""
            <div class='prediction-card'>
                <h2>Predicted Digit: {prediction}</h2>
                <p>Confidence: {confidence:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Clear button
        if st.button("Clear Drawing"):
            st.session_state.canvas_key += 1
            st.rerun()
        
        # Probability chart
        fig = px.bar(
            x=range(10),
            y=probs,
            labels={'x': 'Digit', 'y': 'Probability'},
            color=probs,
            color_continuous_scale='rainbow',
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)

# ========== SIDEBAR ==========
with st.sidebar:
    st.markdown("""
    ## 🎨 Drawing Tips
    - Draw in the center of the canvas
    - Make digits large and clear
    - Use thick strokes for better recognition
    - Click "Clear Drawing" to start over
    """)
    
    st.markdown("---")
    st.markdown("""
    ## ⚙️ Model Info
    - **Algorithm:** K-Nearest Neighbors (KNN)
    - **Training Data:** MNIST dataset
    - **Accuracy:** ~98%
    - **Input Size:** 8×8 pixels (grayscale)
    """)
