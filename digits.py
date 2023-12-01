import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Imports a dataset of 60,000 handwritten digits
mnist = tf.keras.datasets.mnist
# Loads and splits data in a 9:1 ratio
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Scales down x_train and x_test data to 0-1
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# Layers are stacked sequentially, taking the output of the previous one as input
model = tf.keras.models.Sequential()
# Flattens layer into one dimension, expecting an input array of dimensions 28x28 pixels
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
# Denses and connects neurons, with ReLU activation twice - good for image classification
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
# Tries to take all the outputs (10 neurons) and classify them as output
model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))

# Compiling and testing accuracy of the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=3)

# Outputting values of test data (97% accuracy)
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Accuracy: {accuracy}")
print(f"Loss: {loss}")

# Saving the data into the model
model.save("digits.model")

# Loading self-written digit images into the model
for x in range(1,7):
    img = cv.imread(f"{x}.png")[:,:,0]
    # Inverts from white-on-black to black-on-white
    img = np.invert(np.array([img]))
    # Inverted image is passed onto the model, which outputs a probability distribution
    prediction = model.predict(img)
    # The most probable label is chosen and displayed
    print(f"The result is probably: {np.argmax(prediction)}")
    # Visualizes the process in grayscale
    plt.imshow(img[0], cmap=plt.cm.binary)
    # Displays the image on the screen
    plt.show()
