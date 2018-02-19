import numpy as np
import tensorflow as tf
import gzip
import random

NUM_TRAIN_DATA = 60000 #MNIST TRAIN DATA SIZE
NUM_TEST_DATA = 10000 #MNIST TEST DATA SIZE
NUM_EPOCH = 20
NUM_BATCH = 500
PROB = 0.7 #dropout

#read MNIST gzip data format / output = image, label (nparray)
def read_data(if_test): #if_test == 0 : train case / else : test case
  PATH_TRAIN_IMG = '/home/ubuntu/mnist/train-images-idx3-ubyte.gz'
  PATH_TRAIN_LBL = '/home/ubuntu/mnist/train-labels-idx1-ubyte.gz'
  PATH_TEST_IMG = '/home/ubuntu/mnist/t10k-images-idx3-ubyte.gz' 
  PATH_TEST_LBL = '/home/ubuntu/mnist/t10k-labels-idx1-ubyte.gz'
  
  #train case
  if if_test == 0:  
    fp_img = gzip.open(PATH_TRAIN_IMG, 'rb')
    fp_lbl = gzip.open(PATH_TRAIN_LBL, 'rb')
    fp_img.read(16) #not used data
    fp_lbl.read(8) #not used data
    
    train_img = np.zeros(shape=(NUM_TRAIN_DATA, 784)) # (60000, 784)
    train_lbl = np.zeros(shape=(NUM_TRAIN_DATA, 10)) # (60000, 10)
    
    for i in range(NUM_TRAIN_DATA):
      img = np.squeeze(np.frombuffer(fp_img.read(784), np.uint8))
      train_img[i] = img
      
      lbl = np.frombuffer(fp_lbl.read(1), np.uint8)
      lbl_onehot = np.squeeze(np.eye(10)[lbl]) #number -> one-hot format // 2 ->[0,0,1,0,0,0...]
      train_lbl[i] = lbl_onehot
      
    return train_img.astype(dtype = np.float32), train_lbl.astype(dtype = np.float32) #component format : uint8 -> float32
  
  #test case
  else:
    fp_img = gzip.open(PATH_TEST_IMG, 'rb')
    fp_lbl = gzip.open(PATH_TEST_LBL, 'rb')
    fp_img.read(16) #not used data
    fp_lbl.read(8) #not used data
    
    test_img = np.zeros(shape=(NUM_TEST_DATA, 784)) # (10000, 784)
    test_lbl = np.zeros(shape=(NUM_TEST_DATA, 10)) # (10000, 10)
    
    for i in range(NUM_TEST_DATA):
      img = np.squeeze(np.frombuffer(fp_img.read(784), np.uint8))
      test_img[i] = img
      
      lbl = np.frombuffer(fp_lbl.read(1), np.uint8)
      lbl_onehot = np.squeeze(np.eye(10)[lbl]) #number -> one-hot format // 2 ->[0,0,1,0,0,0...]
      test_lbl[i] = lbl_onehot
      
    return test_img.astype(dtype = np.float32), test_lbl.astype(dtype = np.float32) #component format : uint8 -> float32

  
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
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = final, labels = y))
opt = tf.train.AdamOptimizer(1e-3).minimize(loss)
saver = tf.train.Saver()

with tf.Session() as sess:
  init = tf.global_variables_initializer()
  sess.run(init)
  
  xx, yy = read_data(0) #train case
  
  #SGD
  for j in range(NUM_EPOCH):
    avg_err = 0
    idx = np.arrange(0, NUM_TRAIN_DATA)
    np.random.shuffle(idx) #ex:[2831 126 593 2 ...]
    iter = int(NUM_TRAIN_DATA / NUM_BATCH) # ex : 7/3 = 2
    
    fot i in range(iter):
      idx_start = NUM_BATCH*i
      idx_end = idx_start + NUM_BATCH
      ltmp = idx[idx_start:idx_end]
      
      img_shuffle = [
