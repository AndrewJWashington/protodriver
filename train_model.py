import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import initializers
import os

# config
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
batch_size = 50
epochs = 20
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

    # AlexNet
    # https://medium.com/datadriveninvestor/cnn-architecture-series-alexnet-with-implementation-part-ii-7f7afa2ac66a

    model = keras.Sequential()

    # Convolutional layers
    model.add(layers.Conv2D(filters=20, input_shape=input_shape, kernel_size=(11,11), strides=(4,4), padding="valid", activation = "relu"))
    model.add(layers.MaxPool2D(pool_size=(3,3), strides=(2,2), padding="valid"))
    model.add(layers.Conv2D(filters=10, kernel_size=(5,5), strides=(1,1), padding="same", activation = "relu"))
    model.add(layers.MaxPool2D(pool_size=(3,3), strides=(2,2), padding="valid"))
    #model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding="same", activation = "relu"))
    #model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding="same", activation = "relu"))
    #model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="same", activation = "relu"))
    #model.add(MaxPool2D(pool_size=(3,3), strides=(2,2), padding="valid"))

    # Passing it to a Fully Connected layer
    model.add(layers.Flatten())
    # FC Layers
    model.add(layers.Dense(units = 10, activation = "relu", kernel_initializer=initializers.RandomNormal(stddev=0.01)))
    #model.add(layers.Dense(units = 10, activation = "relu", kernel_initializer=initializers.RandomNormal(stddev=0.01)))
    #model.add(layers.Dense(10, activation = "relu", kernel_initializer=initializers.RandomNormal(stddev=0.01)))

    # Output Layer
    model.add(layers.Dense(num_outputs, activation = "sigmoid", kernel_initializer=initializers.RandomNormal(stddev=0.01)))

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

    print("Average predictions:")
    predictions = model.predict(images)
    print(predictions[:,0].mean(), predictions[:,1].mean(), predictions[:,2].mean(), predictions[:,3].mean())
