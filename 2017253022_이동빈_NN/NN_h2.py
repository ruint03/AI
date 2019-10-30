import tensorflow as tf
import numpy as np
from itertools import chain

fp = open("train.txt", "r", encoding="utf-8")
a_x = list()
a_z_vtr = list()
train_all_x = list()
train_all_z = list()

for i in range(60000):
    label_ch = fp.readline()
    label = int(label_ch)
    a_z_vtr = []
    for j in range(10):
        thisbit = 0
        if (j == label):
            thisbit = 1
        a_z_vtr.append(thisbit)
    train_all_z.append(a_z_vtr)

    a_x = []
    for i_x in range(28):
        line_x_string = fp.readline()
        line_x_string_split = line_x_string.split()
        for k in range(len(line_x_string_split)):
            pixel_str = line_x_string_split[k]
            pixel_float = float(pixel_str)
            a_x.append(pixel_float / 255)
    train_all_x.append(a_x)
print("num of train X =", len(train_all_x))
print("num of train Z =", len(train_all_z))
fp.close()
####### Construct the model of NN ############################################

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_nomal([784, 256], stddev=0.01))
L1 = tf.nn.relu(tf.matmul(X, W1))
L1 = tf.nn.dropout(L1, keep_prob)

W2 = tf.Variable(tf.random_nomal([256, 256], stddev=0.01))
L2 = tf.nn.relu(tf.matmul(L1, W2))
L2 = tf.nn.dropout(L2, keep_prob)

W3 = tf.Variable(tf.random_nomal([256, 10], stddev=0.01))
model = tf.matmul(L2, W3)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=model))

optimizer = tf.train.AdamOptimizer(0.0007).minimize(cost)
######### Training the NN Model ##################################################
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

batch_size = 100
total_batch = int(60000 / batch_size)

for epoch in range(60):
    total_cost = 0
    start_example_idx = 0

    for i in range(total_batch):
        batch_xs, batch_ys = [], []
        start_example_idx = batch_size * i
        for ib in range(batch_size):
            batch_xs.append(train_all_x[start_example_idx + ib])
            batch_ys.append(train_all_z[start_example_idx + ib])

        _, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 0.8})

        total_cost += cost_val

    print('Epoch:', '%04d' % (epoch + 1), 'Avarage cost=', total_cost / (total_batch))
print('Finishing Training.')
######### Test the performance of the system ###########################################
fp = open("test.txxt", "r", encoding="utf-8")
test_all_x = list()
test_all_z = list()

for i in range(10000):
    label_ch = fp.readline()
    label = int(label_ch)
    a_z_vtr = []
    for j in range(10):
        thisbit = 0
        if (j == label):
            thisbit = 1
        a_z_vtr.append(thisbit)
    test_all_z.append(a_z_vtr)

    a_x = []
    for i_x in range(28):
        line_x_string = fp.readline()
        line_x_string_split = line_x_string.split()
        for k in range(len(line_x_string_split)):
            pixel_str = line_x_string_split[k]
            pixel_float = float(pixel_str)
            a_x.append(pixel_float / 255)
    test_all_x.append(a_x)
print("num of test X =", len(test_all_x))
print("num of test Z =", len(test_all_z))
fp.close()

is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct,tf.float32))
print('SYSTEM ACCURACY:',sess.run(accuracy, feed_dict={X: test_all_x, Y: test_all_z, keep_prob: 1}))

sess.close()