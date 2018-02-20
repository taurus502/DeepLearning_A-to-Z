#Digit classification using MNIST dataset with fully connected layers only
#Offline coding version(not used MNIST online library)
#1)Download original MNIST dataset files; train-images-idx3..., train-labels-idx1...
#2)Change path

import numpy as np
import tensorflow as tf
import gzip
import random

NUM_TRAIN_DATA = 60000 #NUM OF MNIST TRAIN DATA
NUM_TEST_DATA = 10000 #NUM OF MNIST TEST DATA
NUM_EPOCH = 20
NUM_BATCH = 500
PROB = 0.7 #dropout

#read MNIST gzip data format / output : train image, label, test image, label (nparray)
def read_data(): #for reading raw 'gz'files(ubyte)
  PATH_TRAIN_IMG = '/home/ubuntu/mnist/train-images-idx3-ubyte.gz'
  PATH_TRAIN_LBL = '/home/ubuntu/mnist/train-labels-idx1-ubyte.gz'
  PATH_TEST_IMG = '/home/ubuntu/mnist/t10k-images-idx3-ubyte.gz' 
  PATH_TEST_LBL = '/home/ubuntu/mnist/t10k-labels-idx1-ubyte.gz'
  
  ### MNIST train data set ###
  fp_img1 = gzip.open(PATH_TRAIN_IMG, 'rb')
  fp_lbl1 = gzip.open(PATH_TRAIN_LBL, 'rb')
  fp_img1.read(16) #not used data
  fp_lbl1.read(8) #not used data
    
  train_img = np.zeros(shape=(NUM_TRAIN_DATA, 784)) # (60000, 784)
  train_lbl = np.zeros(shape=(NUM_TRAIN_DATA, 10)) # (60000, 10)
    
  for i in range(NUM_TRAIN_DATA):
    img1 = np.squeeze(np.frombuffer(fp_img1.read(784), np.uint8))
    train_img[i] = img1

    lbl1 = np.frombuffer(fp_lbl1.read(1), np.uint8)
    lbl_onehot1 = np.squeeze(np.eye(10)[lbl1]) #number -> one-hot format // 2 ->[0,0,1,0,0,0...]
    train_lbl[i] = lbl_onehot1
  
  train_npimg = train_img.astype(dtype = np.float32)#component format : uint8 -> float32
  train_nplbl = train_lbl.astype(dtype = np.float32)#component format : uint8 -> float32

  ### MNIST test data set ###
  fp_img2 = gzip.open(PATH_TEST_IMG, 'rb')
  fp_lbl2 = gzip.open(PATH_TEST_LBL, 'rb')
  fp_img2.read(16) #not used data
  fp_lbl2.read(8) #not used data

  test_img = np.zeros(shape=(NUM_TEST_DATA, 784)) # (10000, 784)
  test_lbl = np.zeros(shape=(NUM_TEST_DATA, 10)) # (10000, 10)

  for j in range(NUM_TEST_DATA):
    img2 = np.squeeze(np.frombuffer(fp_img2.read(784), np.uint8))
    test_img[j] = img2

    lbl2 = np.frombuffer(fp_lbl2.read(1), np.uint8)
    lbl_onehot2 = np.squeeze(np.eye(10)[lbl2]) #number -> one-hot format // 2 ->[0,0,1,0,0,0...]
    test_lbl[j] = lbl_onehot2
    
  test_npimg = test_img.astype(dtype = np.float32)
  test_nplbl = test_lbl.astype(dtype = np.float32)
  
  return train_npimg, train_nplbl, test_npimg, test_nplbl
  
  
#define neural network model
def model(inputs, prob):#inputs : img / prob : prob. of dropout
  NUM_LAYER1 = 100
  w1 = tf.Variable(tf.truncated_normal(shape=[784, NUM_LAYER1], stddev = 1e-2))
  b1 = tf.Variable(tf.constant(1e-5, shape=[NUM_LAYER1]))
  l1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(inputs, w1), b1))
  
  NUM_LAYER2 = 40
  w2 = tf.Variable(tf.truncated_normal(shape=[NUM_LAYER1, NUM_LAYER2], stddev = 1e-2))
  b2 = tf.Variable(tf.constant(1e-5, shape=[NUM_LAYER2]))
  l2 = tf.nn.relu(tf.nn.bias_add(tf.matmul(l1, w2), b2))
  l2 = tf.nn.dropout(l2, keep_prob = prob)
  
  w3 = tf.Variable(tf.truncated_normal(shape=[NUM_LAYER2, 10], stddev = 1e-2))
  b3 = tf.Variable(tf.constant(1e-5, shape=[10]))
  l3 = tf.nn.relu(tf.nn.bias_add(tf.matmul(l2, w3), b3))
  
  return l3


#placeholder
x_ = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
hypothesis = model(x_, PROB)

#cost function, optimization, saver
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = hypothesis, labels = y_))
opt = tf.train.AdamOptimizer(1e-3).minimize(loss)
saver = tf.train.Saver()
#save_path = '/mnt/disk3/saved/model.ckpt'

#load data
train_x, train_y, test_x, test_y = read_data()

#session
with tf.Session() as sess:
  init = tf.global_variables_initializer()
  sess.run(init)
    
  #SGD
  for j in range(NUM_EPOCH):
    train_err = 0
    test_err = 0
    idx = np.arrange(0, NUM_TRAIN_DATA)
    np.random.shuffle(idx) #ex:[2831 126 593 2 ...]
    iter = int(NUM_TRAIN_DATA / NUM_BATCH) # ex : 7/3 = 2
    
    #iteration for 1 epoch
    for i in range(iter):
      idx_start = NUM_BATCH*i
      idx_end = idx_start + NUM_BATCH
      ltmp = idx[idx_start:idx_end]
      
      img_shuffle = [train_x[k] for k in ltmp]
      lbl_shuffle = [train_y[k] for k in ltmp]
      
      losstmp, _ = sess.run([loss, opt], feed_dict={x_:img_shuffle, y_:lbl_shuffle, PROB:PROB}) #update variables
      
      loss_ = sess.run(loss, feed_dict={x_:img_shuffle, y_:lbl_shuffle, PROB:1.0}) #measure loss without dropout
      train_err = train_err + loss_
      
    print('train loss at epoch %d ---> %f' %(j, train_err/iter))
    print('test loss ---> %f', sess.run(loss, feed_dict={x_:test_X, y_:test_y, PROB:1.0})
  
  #saver.save(sess, save_path)
      
