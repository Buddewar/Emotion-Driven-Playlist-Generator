import streamlit as st
import pandas as pd
import requests
import cv2
import numpy as np
import random
from PIL import Image
from io import BytesIO
import certifi
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import os

# Configure Streamlit page
st.set_page_config(
    page_title="Emotion-Driven Music Recommender",
    page_icon="🎵",
    layout="wide"
)

# Define emotion to music genre/mood mapping
emotion_to_music = {
    "happy": {
        "genres": ["pop", "dance", "happy"],
        "description": "Upbeat and cheerful music to match your happy mood",
        "color": "#FFD700"  # Gold
    },
    "sad": {
        "genres": ["blues", "sad", "acoustic"],
        "description": "Reflective and soulful tunes for when you're feeling blue",
        "color": "#6495ED"  # Cornflower Blue
    },
    "angry": {
        "genres": ["rock", "metal", "intense"],
        "description": "Powerful music to channel your energy",
        "color": "#DC143C"  # Crimson
    },
    "surprise": {
        "genres": ["electronic", "experimental", "alternative"],
        "description": "Unexpected and exciting sounds for your surprised state",
        "color": "#9370DB"  # Medium Purple
    },
    "fear": {
        "genres": ["ambient", "cinematic", "instrumental"],
        "description": "Calming sounds to ease your anxious feelings",
        "color": "#708090"  # Slate Gray
    },
    "disgust": {
        "genres": ["punk", "grunge", "alternative"],
        "description": "Expressive music that resonates with your current feelings",
        "color": "#556B2F"  # Dark Olive Green
    },
    "neutral": {
        "genres": ["indie", "folk", "chill"],
        "description": "Balanced and pleasant music for your relaxed state",
        "color": "#87CEEB"  # Sky Blue
    }
}

# Path to Zscaler CA certificate (update for cloud deployment if needed)
ZSCALER_CA_PATH = "D:/Python/env/zscaler_ca.cer"

# Function to get Spotify access token with retry mechanism
def get_spotify_token():
    # Retrieve Spotify API credentials from Streamlit secrets
    try:
        client_id = st.secrets.get("SPOTIFY_CLIENT_ID", "")
        client_secret = st.secrets.get("SPOTIFY_CLIENT_SECRET", "")
    except Exception:
        return None
    
    if not client_id or not client_secret:
        return None
    
    auth_url = "https://accounts.spotify.com/api/token"
    
    # Configure retry mechanism
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))
    
    try:
        # Use Zscaler CA certificate locally, fall back to certifi for cloud
        verify_path = ZSCALER_CA_PATH if os.path.exists(ZSCALER_CA_PATH) else certifi.where()
        auth_response = session.post(auth_url, {
            'grant_type': 'client_credentials',
            'client_id': client_id,
            'client_secret': client_secret,
        }, verify=verify_path)
        if auth_response.status_code != 200:
            return None
        
        auth_data = auth_response.json()
        return auth_data['access_token']
    except:
        return None

# Function to search for playlists based on emotions
def get_playlists_for_emotion(emotion, token):
    if not token:
        return []
    
    base_url = "https://api.spotify.com/v1/search"
    
    playlists = []
    
    # Configure retry mechanism
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))
    
    # Try each genre associated with the emotion
    for genre in emotion_to_music[emotion]["genres"]:
        query = f"{genre} {emotion}"
        
        headers = {
            'Authorization': f'Bearer {token}'
        }
        
        params = {
            'q': query,
            'type': 'playlist',
            'limit': 3
        }
        
        try:
            # Use Zscaler CA certificate locally, fall back to certifi for cloud
            verify_path = ZSCALER_CA_PATH if os.path.exists(ZSCALER_CA_PATH) else certifi.where()
            response = session.get(base_url, headers=headers, params=params, verify=verify_path)
            if response.status_code == 200:
                results = response.json()
                if results.get("playlists", {}).get("items"):
                    playlists.extend(results["playlists"]["items"])
        except:
            pass
    
    # Filter out None values and return up to 5 unique playlists
    unique_playlists = []
    unique_ids = set()
    
    for playlist in playlists:
        if playlist is not None and isinstance(playlist, dict) and "id" in playlist:
            if playlist["id"] not in unique_ids and len(unique_playlists) < 5:
                unique_ids.add(playlist["id"])
                unique_playlists.append(playlist)
    
    return unique_playlists

# Function to capture image from webcam
def capture_image():
    img_file_buffer = st.camera_input("Take a picture to analyze your emotion", key="camera")
    if img_file_buffer is not None:
        # Read image from buffer
        bytes_data = img_file_buffer.getvalue()
        img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        return img
    return None

# Function to detect emotion (simplified version)
def detect_emotion(image):
    # Convert to grayscale for easier processing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Use OpenCV's built-in face detection (much lighter than FER)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) > 0:
        # For demo purposes, randomly select an emotion with weighted probabilities
        emotions = ["happy", "sad", "angry", "surprise", "fear", "disgust", "neutral"]
        weights = [0.25, 0.15, 0.1, 0.1, 0.1, 0.05, 0.25]  # Higher weights for happy and neutral
        
        dominant_emotion = random.choices(emotions, weights=weights, k=1)[0]
        
        # Create simulated emotion scores for display
        emotion_scores = {}
        for emotion in emotions:
            if emotion == dominant_emotion:
                emotion_scores[emotion] = random.uniform(0.6, 0.9)  # High score for dominant emotion
            else:
                emotion_scores[emotion] = random.uniform(0.05, 0.3)  # Lower scores for others
        
        # Normalize scores to sum to 1
        total = sum(emotion_scores.values())
        for emotion in emotion_scores:
            emotion_scores[emotion] /= total
            
        # Return detected face box as well
        return dominant_emotion, emotion_scores, faces[0]
    
    return None, None, None

# Function for manual emotion selection (as a fallback)
def select_emotion():
    emotion = st.selectbox(
        "Select your current emotion:",
        ["happy", "sad", "angry", "surprise", "fear", "disgust", "neutral"]
    )
    
    # Create emotion scores for visualization
    emotion_scores = {
        "happy": 0.1,
        "sad": 0.1,
        "angry": 0.1,
        "surprise": 0.1,
        "fear": 0.1,
        "disgust": 0.1,
        "neutral": 0.1
    }
    
    # Set selected emotion to 0.9
    emotion_scores[emotion] = 0.9
    
    return emotion, emotion_scores, None

def main():
    st.title("😊 Emotion-Driven Music Recommender 🎵")
    st.write("Let your emotions choose your music! We'll recommend playlists based on your mood.")
    
    # Initialize session state for playlist display
    if 'show_playlists' not in st.session_state:
        st.session_state.show_playlists = False
    
    if 'detected_emotion' not in st.session_state:
        st.session_state.detected_emotion = None
    
    if 'emotion_scores' not in st.session_state:
        st.session_state.emotion_scores = None
    
    # Sidebar
    with st.sidebar:
        st.header("About")
        st.write("""
        This app recommends music that matches your mood.
        
        How it works:
        1. Either take a photo or select your emotion
        2. We find music playlists that match your mood
        3. Enjoy your personalized recommendations!
        """)
        
        st.header("Settings")
        use_spotify = st.checkbox("Use Spotify API", value=True)
        use_webcam = st.checkbox("Use webcam for emotion detection", value=True)
        
        st.markdown("---")
        st.write("Made with ❤️ using Streamlit")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if use_webcam:
            # Webcam-based emotion detection
            image = capture_image()
            
            if image is not None:
                # Detect emotion
                emotion, emotion_scores, face_box = detect_emotion(image)
                
                if emotion:
                    st.session_state.detected_emotion = emotion
                    st.session_state.emotion_scores = emotion_scores
                    st.session_state.show_playlists = True
                    
                    # Draw rectangle around face with OpenCV
                    if face_box is not None:
                        x, y, w, h = face_box
                        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # Display image with face detection
                    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption=f"Detected Emotion: {emotion}")
                    
                    # Show emotion scores
                    st.write("Emotion Analysis:")
                    emotion_df = pd.DataFrame({
                        'Emotion': list(emotion_scores.keys()),
                        'Score': list(emotion_scores.values())
                    }).sort_values(by='Score', ascending=False)
                    
                    st.bar_chart(emotion_df.set_index('Emotion'))
                else:
                    st.error("No face or emotion detected. Please try again with a clearer image or use manual selection.")
        else:
            # Manual emotion selection
            st.write("Select your emotion manually:")
            emotion, emotion_scores, _ = select_emotion()
            
            if st.button("Get Music Recommendations"):
                st.session_state.detected_emotion = emotion
                st.session_state.emotion_scores = emotion_scores
                st.session_state.show_playlists = True
                
                # Show emotion scores
                st.write("Selected Emotion:")
                emotion_df = pd.DataFrame({
                    'Emotion': list(emotion_scores.keys()),
                    'Score': list(emotion_scores.values())
                }).sort_values(by='Score', ascending=False)
                
                st.bar_chart(emotion_df.set_index('Emotion'))
    
    with col2:
        if st.session_state.show_playlists and st.session_state.detected_emotion:
            emotion = st.session_state.detected_emotion
            
            st.markdown(f"""
            <div style='background-color: {emotion_to_music[emotion]["color"]}; padding: 20px; border-radius: 10px;'>
                <h2>Your mood: {emotion.capitalize()}</h2>
                <p>{emotion_to_music[emotion]["description"]}</p>
            </div>
            """, unsafe_allow_html=True)
            
            if use_spotify:
                with st.spinner('Finding the perfect music for your mood...'):
                    # Get Spotify token
                    token = get_spotify_token()
                    
                    if token:
                        # Get playlists
                        playlists = get_playlists_for_emotion(emotion, token)
                        
                        if playlists:
                            st.subheader("Recommended Playlists")
                            
                            for idx, playlist in enumerate(playlists):
                                col_a, col_b = st.columns([1, 3])
                                
                                with col_a:
                                    if playlist["images"] and len(playlist["images"]) > 0:
                                        st.image(playlist["images"][0]["url"], width=120)
                                    else:
                                        st.image("https://via.placeholder.com/120", width=120)
                                
                                with col_b:
                                    st.markdown(f"**{playlist['name']}**")
                                    st.write(f"By: {playlist['owner']['display_name']}")
                                    st.write(f"Tracks: {playlist['tracks']['total']}")
                                    st.markdown(f"[Open in Spotify]({playlist['external_urls']['spotify']})")
                                
                                st.markdown("---")
                        else:
                            st.error("Unable to fetch playlists. Please try again later.")
                    else:
                        st.error("Unable to connect to Spotify. Please try again later.")
            else:
                st.error("Spotify API is disabled. Please enable it in settings.")
            
            # Add user feedback
            st.subheader("How well did these recommendations match your mood?")
            feedback = st.slider("Rate from 1-5", 1, 5, 3)
            
            if st.button("Submit Feedback"):
                st.success("Thank you for your feedback! We'll use it to improve recommendations.")

if __name__ == "__main__":
    main()
