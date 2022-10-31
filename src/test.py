import tensorflow as tf

mnist = tf.keras.datasets.mnist

# Split up test data and testing data
(x_test, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the data
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.load_model('../handwritten.model')

loss, accuracy = model.evaluate(x_test, y_test)

print(loss)
print(accuracy)