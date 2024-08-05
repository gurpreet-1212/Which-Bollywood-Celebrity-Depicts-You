import streamlit as st
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import pickle
import os
import numpy as np
from mtcnn import MTCNN
import cv2
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization

# Ensure the uploads directory exists
if not os.path.exists('uploads'):
    os.makedirs('uploads')

# Load the pre-trained EfficientNetB0 model and data
base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze the base model

model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))  # Regularization to prevent overfitting
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))  # Regularization to prevent overfitting
model.add(Dense(128, activation='relu'))  # Output features of dimension 128

# Load feature list and filenames
feature_list = pickle.load(open('embedding.pkl', 'rb'))
filenames = pickle.load(open('filenames.pkl', 'rb'))

detector = MTCNN()

def save_uploaded_image(uploaded_image):
    try:
        upload_path = os.path.join('uploads', uploaded_image.name)
        with open(upload_path, 'wb') as f:
            f.write(uploaded_image.getbuffer())
        return upload_path
    except IOError as e:
        st.error(f"Error saving the uploaded image: {e}")
        return None

def extract_features(img_path, model, detector):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    results = detector.detect_faces(img_rgb)

    if not results:
        st.error("No face detected in the image.")
        return None

    x, y, width, height = results[0]['box']
    face = img_rgb[y:y + height, x:x + width]

    # Resize the face image to the model's expected input size
    image_resized = Image.fromarray(face)
    image_resized = image_resized.resize((224, 224))

    face_array = np.asarray(image_resized)
    face_array = face_array.astype('float32')

    expanded_img = np.expand_dims(face_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img)
    result = model.predict(preprocessed_img).flatten()
    return result

def recommend(feature_list, features):
    if features is None:
        return None
    similarity = []
    for i in range(len(feature_list)):
        similarity.append(cosine_similarity(features.reshape(1, -1), feature_list[i].reshape(1, -1))[0][0])

    index_pos = sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][0]
    return index_pos

st.title('Which Bollywood Celebrity Depicts You?')

uploaded_image = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])

if uploaded_image is not None:
    # Save the uploaded image
    img_path = save_uploaded_image(uploaded_image)
    if img_path:
        # Load the image for display
        display_image = Image.open(img_path)

        # Extract features from the uploaded image
        features = extract_features(img_path, model, detector)
        if features is not None:
            # Recommend the most similar celebrity
            index_pos = recommend(feature_list, features)
            if index_pos is not None:
                predicted_actor = " ".join(filenames[index_pos].split('\\')[1].split('_'))
                # Display results
                col1, col2 = st.columns(2)

                with col1:
                    st.header('Your Uploaded Image')
                    st.image(display_image)

                with col2:
                    st.header("Seems like " + predicted_actor)
                    st.image(filenames[index_pos], width=300)
            else:
                st.error("Failed to find a matching celebrity.")
        else:
            st.error("Failed to extract features from the image.")


