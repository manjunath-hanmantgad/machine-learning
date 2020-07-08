import tensorflow as tf
from tensorflow import keras

import numpy as np 
import matplotlib.pyplot as plt 

print(tf.__version__)

# import the dataset 

fashion_mnist = keras.datasets.fashion_mnist.load_data() 
(train_images , train_labels) , (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top' , 'Trouser', 'Pullover', 'Dress', 'Coat' , 'Sandal' , 'Shirt' , 'Sneaker' , 'Bag', 'Ankle boot']

# explore data 

train_images.shape # gets shape of data 
len(train_labels)

test_images.shape # test shape 
len(test_labels)

# preprocess data 

plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

# scaling images 

train_images = train_images / 255.0
test_images = test_images / 255.0

# display the image with their names below 

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i] , cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

# bulding the model 

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation= 'relu'),
    keras.layers.Dense(10)
])

# compling the model

model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])


# train the model 

# feed the model 

model.fit(train_images, train_labels , epochs=10)

# accuracy check 

test_loss, test_accuracy = model.evaluate(test_images, test_labels , verbose=2)
print('\nTest accuracy:' , test_accuracy)

# now make predictions 

probability_model = tf.keras.Sequential([model,tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)

# check predcitions 
predictions[0]

np.argmax(predictions[0])
test_labels[0] # to check the prediction 

# plotting this

def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array, true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predcited_label],100*np.max(predictions_array),class_names[true_label]), color=color)

def plot_value_array(i, predictions_array, true_label):
    predcitions_array, true_label = predictions_array, true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0,1])
    predicted_label = np.argmax(predcitions_array)

    thisplot[predcited_label].set_color('red')
    thisplot[true_label].set_color('blue')

# verifying the predictions 

i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i], test_labels)
plt.show()

# plotting many images 

num_rows = 5
num_cols = 3
num_images = num_rows * num_cols 
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions[i], test_labels, test_images)
    plot.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()


# using the trained model 

img = test_images[4]
print(img.shape)
predictions_single = probability_model.predict(img)
print(predcitions_single)
plot_value_array(1,predictions_single[0], test_labels_ = plt.xticks(range(10), class_names, rotation=45))

np.argmax(predictions_single[4])