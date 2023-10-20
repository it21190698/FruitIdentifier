import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import streamlit as st
import joblib
import time
import cv2
from PIL import Image
from keras.models import load_model, Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense
from skimage import feature
import matplotlib.pyplot as plt  


IMAGE_WIDTH = 100
IMAGE_HEIGHT = 100
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS = 3
SIZE = 128

class CNNImageClassifier:
    def __init__(self):
        self.cnn_model = None

    def load_model(self):
        self.cnn_model = tf.keras.models.load_model('multiclass_fruit.h5')

    def load_and_preprocess_image(self, img):
        img = img.resize(IMAGE_SIZE, Image.LANCZOS)
        img = image.img_to_array(img)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        return img

    def predict_image_class(self, image):
        class_labels = {
            'freshbanana': 0,
            'rottenbanana': 1,
            'freshapples': 2,
            'rottenapples': 3,
            'freshoranges': 4,
            'rottenoranges': 5
        }
        try:
            predictions = self.cnn_model.predict(image)
            class_index = np.argmax(predictions)
            predicted_class = [k for k, v in class_labels.items() if v == class_index][0]
            return predicted_class
        except Exception as e:
            st.error(f"Error making predictions: {e}")
            return "Unknown"

    def process_image(self, img):
        try:
            img = cv2.imread(img)

            if img is None:
                st.error("Error loading the image. Please make sure it's a valid image file.")
                return None, None, None, None

            img = cv2.resize(img, IMAGE_SIZE)

            if img.shape[-1] != IMAGE_CHANNELS:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            shear_matrix = np.array([[1, 0.2, 0], [0, 1, 0], [0, 0, 1]])
            img_sheared = cv2.warpPerspective(img, shear_matrix, IMAGE_SIZE)

            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_inverted = 255 - img_gray
            img_edge = cv2.Canny(img_gray, 100, 200)

            return img_sheared, img_gray, img_edge, img_inverted
        except Exception as e:
            st.error(f"Error processing the image: {e}")
            return None, None, None, None

    def run(self):
        st.title("CNN Fruit Image Classifier")

        self.load_model()

        st.subheader("Upload an Image for Classification")
        uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

        if uploaded_image is not None:
            try:
                file_details = {"FileName": uploaded_image.name, "FileType": uploaded_image.type, "FileSize": uploaded_image.size}
                st.write(file_details)

                with open(os.path.join("tempDir", uploaded_image.name), "wb") as f:
                    f.write(uploaded_image.getbuffer())

                original_img_path = os.path.join("tempDir", uploaded_image.name)
                original_img = Image.open(original_img_path)

                st.image(original_img, caption="Original Image", use_column_width=True)
                st.write("Processing...")

                with st.empty():
                    self.simulate_processing_time()
                    img_data = Image.open(original_img_path)
                    img_array = np.array(img_data)
                    img = img_array
                    img = self.load_and_preprocess_image(original_img)
                    predicted_class = self.predict_image_class(img)


                processed_imgs = self.process_image(original_img_path)

                if processed_imgs and len(processed_imgs) > 0:
                    num_cols = 2  
                    num_rows = (len(processed_imgs) + num_cols - 1) // num_cols  

                    st.subheader("Processed Images")
                    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 10))
                    for i in range(len(processed_imgs)):
                        ax = axes[i // num_cols, i % num_cols]
                        ax.imshow(processed_imgs[i])
                        ax.axis('off')
                        ax.set_title(f"Processed Image {i + 1}")

                    for i in range(len(processed_imgs), num_rows * num_cols):
                        fig.delaxes(axes.flatten()[i])

                    st.pyplot(fig)
                st.subheader("Prediction")
                st.write(f"Predicted Class: {predicted_class}")
            except Exception as e:
                st.error(f"Error processing the image: {e}")

    def simulate_processing_time(self):
        time.sleep(2)

class RandomForestClassifier:
    def __init__(self):
        self.cnn_model = load_model('cnn_model.h5')

    def predict_image(self, image):
    
        img = np.array(image)
        img = cv2.resize(img, (SIZE, SIZE))
        img = img / 255.0

   
        feature_extractor = Sequential()
        feature_extractor.add(Conv2D(32, 3, activation='relu', input_shape=(SIZE, SIZE, 3)))
        feature_extractor.add(BatchNormalization())
        feature_extractor.add(Conv2D(32, 3, activation='relu'))
        feature_extractor.add(BatchNormalization())
        feature_extractor.add(MaxPooling2D(pool_size=3))
        feature_extractor.add(Conv2D(64, 3, activation='relu'))
        feature_extractor.add(BatchNormalization())
        feature_extractor.add(Conv2D(64, 3, activation='relu'))
        feature_extractor.add(BatchNormalization())
        feature_extractor.add(MaxPooling2D(pool_size=3))
        feature_extractor.add(Flatten())
        img_features = feature_extractor.predict(np.expand_dims(img, axis=0))


        with open('finalized_RF_model.pkl', 'rb') as file:
            RF_model = joblib.load(file)


        predicted_label_encoded = RF_model.predict(img_features)


        class_labels = {
            'freshbanana': 0,
            'rottenbanana': 1,
            'freshapples': 2,
            'rottenapples': 3,
            'freshoranges': 4,
            'rottenoranges': 5
        }


        predicted_label = next(key for key, value in class_labels.items() if value == predicted_label_encoded[0])

        return predicted_label

    def run(self):
        st.title("Random Forest Classifier")

        uploaded_image = st.file_uploader("Upload a fruit image (JPG only):", type=["jpg"])

        if uploaded_image is not None:

            pil_image = Image.open(uploaded_image)

            pil_image = pil_image.convert('RGB')


            pil_image = pil_image.resize((SIZE, SIZE))


            image = np.array(pil_image)

            st.image(image, caption="Uploaded Image", use_column_width=True)

            if st.button("Predict"):
                prediction = self.predict_image(image)
                st.write(f"Predicted Label: {prediction}")

        st.write("Instructions:")
        st.write("1. Upload an image of a fruit (JPG only).")
        st.write("2. Click the 'Predict' button to classify the fruit as fresh or rotten.")

class SVMImageClassifier:
    def __init__(self):
        self.model = None

    def load_model(self):
        self.model = joblib.load('svm_fruit_classifier.h5')

    def preprocess_image(self, img):
        try:
            img = Image.open(img)
            img = img.resize(IMAGE_SIZE, Image.LANCZOS)
            img = np.array(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, IMAGE_SIZE)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = 255 - img
            img = cv2.Canny(img, 100, 200)
            feature_vector = feature.hog(img)
            return feature_vector
        except Exception as e:
            st.error(f"Error processing the image: {e}")
            return None

    def predict_image_class(self, feature_vector):
        class_labels = {
            'freshapples': 0,
            'freshbanana': 1,
            'freshoranges': 2,
            'rottenapples': 3,
            'rottenbanana': 4,
            'rottenoranges': 5
        }
        try:
            predicted_class = self.model.predict([feature_vector])[0]
            return predicted_class  
        except Exception as e:
            st.error(f"Error making predictions: {e}")
            return -1  

    def run(self):
        st.title("SVM Fruit Image Classifier")

        self.load_model()

        st.subheader("Upload an Image for Classification")
        uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

        if uploaded_image is not None:
            try:
                file_details = {"FileName": uploaded_image.name, "FileType": uploaded_image.type, "FileSize": uploaded_image.size}
                st.write(file_details)

                with open(os.path.join("tempDir", uploaded_image.name), "wb") as f:
                    f.write(uploaded_image.getbuffer())

                original_img_path = os.path.join("tempDir", uploaded_image.name)

                st.image(original_img_path, caption="Original Image", use_column_width=True)
                st.write("Processing...")

                with st.empty():
                    self.simulate_processing_time()
                    img = original_img_path
                    feature_vector = self.preprocess_image(img)
                    if feature_vector:
                        predicted_class = self.predict_image_class(feature_vector)
                        if predicted_class != -1:
                            class_labels = {
                                0: 'freshapples',
                                1: 'freshbanana',
                                2: 'freshoranges',
                                3: 'rottenapples',
                                4: 'rottenbanana',
                                5: 'rottenoranges'
                            }
                            predicted_fruit = class_labels[predicted_class]
                            st.subheader("Prediction")
                            st.write(f"Predicted Class: {predicted_fruit}")
                        else:
                            st.error("Unknown or error in prediction")

            except Exception as e:
                st.error(f"Error processing the image: {e}")

    def simulate_processing_time(self):
        time.sleep(2)

def main():
    st.title("Fresh and Rotten Fruits Classification")

    model_choice = st.radio("Choose Model", ["CNN Image Classifier", "Random Forest Classifier", "SVM Image Classifier"])

    if model_choice == "CNN Image Classifier":
        cnn_classifier = CNNImageClassifier()
        cnn_classifier.run()
    elif model_choice == "Random Forest Classifier":
        fresh_rotten_classifier = RandomForestClassifier()
        fresh_rotten_classifier.run()
    elif model_choice == "SVM Image Classifier":
        svm_classifier = SVMImageClassifier()
        svm_classifier.run()

if __name__ == "__main__":
    main()
