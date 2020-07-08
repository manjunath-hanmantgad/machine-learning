import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# normalize the input

def normalize(X):
    mean = np.mean(X)
    std = np.std(X)
    X = (X - mean) / std
    return X

# load the data

boston = tf.contrib.learn.datasets.load_dataset('boston')
X_train , Y_train = boston.data[:,5] , boston.target
n_samples = len(X_train)

#placeholders

X = tf.placeholder(tf.float32, name ='X')
Y = tf.placeholder(tf.float32, name ='Y') 

# variables for weight(w) and bias(b)

b = tf.Variable(0.0)
w = tf.Variable(0.0)

# linear regression model

Y_pred = X * w + b

# loss function 

lossfunction = tf.sqaure(Y - Y_pred, name = 'lossfunction')

# optimizer 

optimizer = tf.train.GradientDescentOptimizer(learning_rate= 0.04).minimize(lossfunction)

# initialize 

init_op = tf.gloabl_variables_initializer()
total = []

# train model 

with tf.session() as sess:
    sess.run(init_op)
    writer = tf.summary.Filewriter('graphical', sess.graphical)
    for i in range(100):
        total_loss = 0
        for x,y in zip(X_train,Y_train):
            _, l = sess.run([optimizer, lossfunction], feed_dict={X:x, Y:y})
            total_loss += 1
        total.append(total_loss / n_samples)
        print('Epoch {0} : Loss {1} '.format(i, total_loss / n_samples))
        writer.close()
        b_value , w_value = sess.run([b,w])

Y_output = X_train * w_value + b_value
print('Done')

# plot the result 

plt.plot(X_train, Y_train , 'bo', label = 'Real Data')
plt.plot(X_train, Y_output, 'r' , label = 'Predicted Data')
plt.legend()
plt.show()
plt.plot(total)
plt.show()

"""
DATA_FILE= 'boston_housing.csv'
BATCH_SIZE= 10
NUM_FEATURES

def data_generator(filename):
    f_queue = tf.train.string_input_producer(filename)
    reader = tf.TextLineReader(skip_headder_lines=1)
    line
    _, value = reader.read(f_queue)
    record_defaults = [0.0] for _ in range(NUM_FEATURES)]
    data = tf.decode_csv(value,record_defaults=record_defaults)
    features = tf.stack(tf.gather_nd(data,[[5],[10],[12]]))
    label = data[-1]
    dequeuemin_after_dequeue = 10 * BATCH_SIZE
    capacity = 20 * BATCH_SIZE
    feature_batch, label_batch = tf.train.shuffle_batch([features,label], batch_size=BATCH_SIZE, min_after_dequeue=min_after_dequeue)
    return feature_batch , label_batch

def generate_data(feature_batch, label_batch):
    with tf.session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for _ in range(5):
            features,labels = sess.run([feature_batch, label_batch])
            print(features,"HI")
        coord.request_stop()
        coord.join(threads)
    
if__name__ == '__main__' :
    feature_batch, label_batch = data_generator([DATA_FILE])
    generate_data(feature_batch,label_batch)
"""

