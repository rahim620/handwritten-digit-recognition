import tensorflow as tf

mnist = tf.keras.datasets.mnist

# Split up training data and testing data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the data
x_train = tf.keras.utils.normalize(x_train, axis=1)

# Sequential neural network
model = tf.keras.models.Sequential()

# Turn the 28x28 grid of pixels into one line of 784 pixels
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))

# 3 layers where each neuron of one layer is connected to each neuron of the next layer, last layer is the output
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=3)

model.save('handwritten.model')


