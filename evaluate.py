import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

IMAGE_WIDTH = 100
IMAGE_HEIGHT = 100
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS = 3

model = tf.keras.models.load_model('multiclass_fruit.h5')

dataset_directory = '/kaggle/input/fruits-fresh-and-rotten-for-classification/dataset/train'

class_labels = {
    'freshbanana': 0,
    'rottenbanana': 1,
    'freshapples': 2,
    'rottenapples': 3,
    'freshoranges': 4,
    'rottenoranges': 5
}

num_images_per_class = 5

true_labels = []
predicted_labels = []

for label in class_labels.keys():
    class_directory = os.path.join(dataset_directory, label)
    image_files = os.listdir(class_directory)
    random.shuffle(image_files)
    selected_images = image_files[:num_images_per_class]
    
    for image_file in selected_images:
        img = image.load_img(os.path.join(class_directory, image_file), target_size=(100, 100))
        img = image.img_to_array(img)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        predictions = model.predict(img)
        class_index = np.argmax(predictions)

        predicted_class = [k for k, v in class_labels.items() if v == class_index][0]

        true_labels.append(label)
        predicted_labels.append(predicted_class)

labels_list = list(class_labels.keys())

confusion = confusion_matrix(true_labels, predicted_labels, labels=labels_list)

display = ConfusionMatrixDisplay(confusion, display_labels=labels_list)
display.plot(cmap=plt.cm.Blues, values_format='d')

plt.show()

classification_report_output = classification_report(true_labels, predicted_labels, target_names=labels_list)

print("Classification Report:\n", classification_report_output)

accuracy = (confusion.diagonal() / confusion.sum(axis=1)).mean()
print("Overall Accuracy:", accuracy)
