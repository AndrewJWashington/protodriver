import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

# config
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
batch_size = 15
epochs = 10
MAX_SESSIONS = 20

num_outputs = 4 # gas, left, brake, right
input_shape = (75, 100, 3) # 75, 100 rgb

# big help
# https://www.pyimagesearch.com/2018/06/04/keras-multiple-outputs-and-multiple-losses/


if __name__ == "__main__":
    # read training data
    for session_number in range(1, MAX_SESSIONS):
        if session_number >= MAX_SESSIONS:
            print(f"Warning: You might be exceeding {MAX_SESSIONS} training sessions. " +
                  f"Raise MAX_SESSIONS if you want to train on more sessions.")
        TRAINING_SESSION_ID = "session_" + str(session_number)
        try:
            with open(f'training_data/training_data_{TRAINING_SESSION_ID}.npy', 'rb') as f:
                if 'images' in locals():
                    new_images = np.load(f)
                    images = np.concatenate([images, new_images])
                else:
                    images = np.load(f)
                    
            with open(f'training_data/training_labels_{TRAINING_SESSION_ID}.npy', 'rb') as f:
                if 'labels' in locals():
                    new_labels = np.load(f)
                    labels = np.concatenate([labels, new_labels])
                else:
                    labels = np.load(f)
        except FileNotFoundError:
            print(f"Loaded {session_number - 1} files of training data")
            break

    if 'images' not in locals():
        raise Exception("No training data in training_data/")

    # reshape x
    #images = np.expand_dims(images, -1)

    # reshape y
    # convert class vectors to binary class matrices
    #labels = keras.utils.to_categorical(labels, num_classes)

    # shuffle data (todo)
    # eventually should do shuffle by segments (e.g. turns)

    # test/train split
    training_samples = int(np.ceil(0.9*len(images))) 
    x_train = images[:training_samples]
    x_test =  images[training_samples:]
    y_train = labels[:training_samples]
    y_test = labels[training_samples:]
    
    print("x_train shape:", x_train.shape)
    print("y_train shape:", y_train.shape)
    print("x_test shape:", x_test.shape)
    print("y_test shape:", y_test.shape)

    print("y_train examples:")
    print(y_train[:5])
    
    model = keras.Sequential([
        keras.Input(shape=input_shape),
        layers.Conv2D(3, kernel_size=(10, 10), activation="relu"),
        layers.Flatten(),
        layers.Dropout(0.2),
        layers.Dense(num_outputs, activation="sigmoid"),
        layers.Dense(num_outputs, activation="sigmoid"),
    ])
    model.summary()

    print("Compiling model...")
    model.compile(loss="categorical_crossentropy", optimizer="adam",
                  metrics=["accuracy"])

    print("Training model...")
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
              validation_split=0.1)
    
    model_filename = 'basic_cnn'
    model.save(f'models/{model_filename}')
    print(f"Saved model to models/{model_filename}")

    print("Examples predictions:")
    for prediction in model.predict(x_test[::int(len(x_test)/10)]):
        print(prediction)
