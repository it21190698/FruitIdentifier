import numpy as np
import os
import cv2
from random import shuffle
from keras.utils import to_categorical
from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import wandb

wandb.login(key="")
wandb.init(project="fruit_classification_hyperparameter_tuning", config={})

def load_multiclass_data():
    fruit_categories =['freshbanana', 'rottenbanana', 'freshapples', 'rottenapples', 'freshoranges', 'rottenoranges']
    X, Y = [], []
    data = []

    for category in tqdm(fruit_categories):
        category_path = os.path.join('/kaggle/input/fruits-fresh-and-rotten-for-classification/dataset/train', category)
        for img_name in os.listdir(category_path):
            img = cv2.imread(os.path.join(category_path, img_name))
            img = cv2.resize(img, (100, 100))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            data.append([img, fruit_categories.index(category)])

    shuffle(data)
    for image, label in tqdm(data):
        X.append(image)
        Y.append(label)
    return X, Y

X, Y = load_multiclass_data()

X = np.array(X) / 255.0
Y = to_categorical(Y)

def build_model(hp):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(hp.Int('conv1_units', min_value=32, max_value=256, step=32),
                                     (3, 3), activation='relu', input_shape=(100, 100, 3))
    )
    model.add(tf.keras.layers.MaxPooling2D((2, 2))
    )
    model.add(tf.keras.layers.Dropout(hp.Float('dropout_1', min_value=0.2, max_value=0.5, step=0.1))
    )

    for i in range(hp.Int('num_conv_layers', 1, 3)):
        model.add(tf.keras.layers.Conv2D(hp.Int(f'conv_{i}_units', min_value=32, max_value=256, step=32), (3, 3), activation='relu')
        )
        model.add(tf.keras.layers.MaxPooling2D((2, 2))
        )
        model.add(tf.keras.layers.Dropout(hp.Float(f'dropout_{i}', min_value=0.2, max_value=0.5, step=0.1)
        ))

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(hp.Int('dense_units', min_value=64, max_value=512, step=64), activation='relu')
    )
    model.add(tf.keras.layers.Dropout(hp.Float('dropout_2', min_value=0.2, max_value=0.5, step=0.1)
    ))

    model.add(tf.keras.layers.Dense(6, activation='softmax'))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2))
    ,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=10,
    directory='/kaggle/working',
    project_name='fruit_multiclass_hyperparameter_tuning')

tuner.search(X_train, Y_train, validation_data=(X_val, Y_val), epochs=40, callbacks=[wandb.keras.WandbCallback()])

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

best_model = tuner.hypermodel.build(best_hps)
best_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=best_hps.get('learning_rate')),
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])

best_model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=40)

best_model.save('multiclass_fruit.h5')
