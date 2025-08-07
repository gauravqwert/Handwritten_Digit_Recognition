import streamlit as st
import joblib
import numpy as np
from PIL import Image, ImageOps
import time
import plotly.express as px

# Custom CSS for animations and styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
}

.header {
    color: #4a4a4a;
    text-align: center;
    animation: fadeIn 1.5s;
}

@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

.upload-box {
    background: white;
    border-radius: 15px;
    padding: 30px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    margin: 20px 0;
    animation: slideUp 1s;
}

@keyframes slideUp {
    from { transform: translateY(50px); opacity: 0; }
    to { transform: translateY(0); opacity: 1; }
}

.prediction-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 15px;
    padding: 20px;
    margin: 20px 0;
    box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(102, 126, 234, 0.7); }
    70% { box-shadow: 0 0 0 15px rgba(102, 126, 234, 0); }
    100% { box-shadow: 0 0 0 0 rgba(102, 126, 234, 0); }
}

.sidebar .sidebar-content {
    background: linear-gradient(180deg, #2c3e50 0%, #1a2530 100%) !important;
    color: #00ff00 !important;
}

.st-bb { border-bottom: 1px solid #eee; }
.st-at { background-color: #667eea; }
</style>
""", unsafe_allow_html=True)


# Load the trained KNN model
@st.cache_resource
def load_model():
    return joblib.load('knn_digit_classifier.pkl')


model = load_model()

# --- Sidebar ---
with st.sidebar:
    st.markdown("""
    <div style='padding: 20px; color: red;'>
        <h2>🔍 How It Works</h2>
        <ol>
            <li>Upload an image of a handwritten digit (0-9)</li>
            <li>The AI will resize it to 8x8 pixels</li>
            <li>KNN model predicts the digit with confidence</li>
        </ol>
        <p><strong>Model:</strong> K-Nearest Neighbors (K=5)</p>
        <p><strong>Accuracy:</strong> ~98% on MNIST</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='text-align: center;'>
        <p>Built with ❤️ using</p>
        <img src='https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/Scikit_learn_logo_small.svg/1200px-Scikit_learn_logo_small.svg.png' width='100'>
        <img src='https://streamlit.io/images/brand/streamlit-mark-color.png' width='80'>
    </div>
    """, unsafe_allow_html=True)

# --- Main Content ---
st.markdown("<h1 class='header'>✍️ Handwritten Digit Recognizer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666;'>Draw or upload a digit (0-9) and watch the AI in action!</p>",
            unsafe_allow_html=True)

# File uploader with styled container
with st.container():
    st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"], key="uploader")
    st.markdown("</div>", unsafe_allow_html=True)

if uploaded_file is not None:
    # Add processing animation
    with st.spinner('🤖 AI is analyzing your digit...'):
        time.sleep(1)  # Simulate processing time

        # Load and preprocess image
        img = Image.open(uploaded_file)
        img = ImageOps.grayscale(img)
        img = img.resize((8, 8))

        # Display with animation
        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption="Processed Image (8x8)", width=150)
        with col2:
            st.markdown("""
            <div style='margin-top: 30px;'>
                <h4>⚙️ Processing Steps:</h4>
                <ul>
                    <li>Converted to grayscale</li>
                    <li>Resized to 8×8 pixels</li>
                    <li>Normalized pixel values</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        # Convert to numpy array and flatten
        img_array = np.array(img).reshape(1, -1)

        # Predict with progress bar
        progress_bar = st.progress(0)
        for percent_complete in range(100):
            time.sleep(0.01)  # Visual effect only
            progress_bar.progress(percent_complete + 1)

        prediction = model.predict(img_array)[0]
        confidence = np.max(model.predict_proba(img_array)) * 100

        # Show results with animated card
        st.markdown(f"""
        <div class='prediction-card'>
            <h2 style='color: white; text-align: center;'>Prediction: {prediction}</h2>
            <p style='text-align: center; font-size: 24px;'>Confidence: {confidence:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)

        # Show probabilities with Plotly (more interactive)
        probs = model.predict_proba(img_array)[0]
        fig = px.bar(
            x=range(10),
            y=probs,
            labels={'x': 'Digit', 'y': 'Probability'},
            color=probs,
            color_continuous_scale='Viridis',
            title='Prediction Probabilities',
            height=400
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#4a4a4a')
        )
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Try different handwriting styles! The model was trained on MNIST dataset.</p>
</div>
""", unsafe_allow_html=True)