#Instantly detect digits I drew
#Trained with MNIST dataset
#MUST Load pre-trained MNIST classification network
#MUST Match the model with pre-trained network

#1)Draw Digit(0-9)
#2)Push 'esc'
#3)Digit drawn is shown as 28x28 size
#4)Print probability of the digit


import cv2
import tensorflow as tf
import random
import numpy as np


def onmouse(event, x, y, flags, param):
  # 'flags' is useless
  if event == cv2.EVENT_LBUTTONDOWN:
    grad = random.random() * 0.3 + 0.7 #brightness of points with randomness
    cv2.circle(param, (x, y), 15, (255*grad, 255*grad, 255*grad), -1)
    #radius == 15 / default color == (255, 255, 255) which means white color

    
    
def mousebrush():
  global drawn_data
  
  scale = 20
  img = np.zeros([28*scale, 28*scale, 3], np.uint8)
  cv2.namedWindow('paint')
  cv2.setMouseCallback('paint', onmouse, param = img)
  
  while True:
    cv2.imshow('paint', img)
    k = cv2.waitKey(1) & 0xFF
    
    if k == 27: #button means 'esc'
      tmp = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
      new_img = cv2.resize(tmp, (28, 28))
      
      while True:
        cv2.imshow('paint', new_img)
        k = cv2.waitKey(1) & 0xFF
        
        if k == 27: #button means 'esc'
          drawn_data = np.asarray(new_img).flatten()
          #asarray : tuple -> nparray
          #flatten : 2d nparray(28,28) -> 1d nparray(784,)
          break
      break
      
  cv2.destroyAllWindows()
  
  
  
def customize_test(data):
  prob = 1
  
  def model(inputs):
    num_layer1 = 100
    w1 = tf.Variable(tf.truncated_normal(shape=[784, num_layer1], stddev = 0.01))
    b1 = tf.Variable(tf.constant(1e-5, shape=[num_layer1]))
    l1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(inputs, w1), b1))
    
    num_layer2 = 40
    w2 = tf.Variable(tf.truncated_normal(shape=[num_layer1, num_layer2], stddev = 0.01))
    b2 = tf.Variable(tf.constant(1e-5, shape=[num_layer2]))
    l2 = tf.nn.relu(tf.nn.bias_add(tf.matmul(l1, w2), b2))
    l2 = tf.nn.dropout(l2, keep_prob = prob)
    
    w3 = tf.Variable(tf.truncated_normal(shape=[num_layer2, 10], stddev = 0.01))
    b3 = tf.Variable(tf.constant(1e-5, shape=[10]))
    l3 = tf.nn.relu(tf.nn.bias_add(tf.matmul(l2, w3), b3))
    
    return l3
  
  x = tf.placeholder(tf.float32, [None, 784])
  #y = tf.placeholder(tf.float32, [None, 10])
  final = model(x)

  #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = final, labels = y))
  #opt = tf.train.AdamOptimizer(0.001).minimize(loss)
  
  init = tf.global_variables_initializer()
  saver = tf.train.Saver()
  path = '/mnt/disk2/saved_mnist/model2/model_new.ckpt' #file path
  
  with tf.Session() as sess:
    sess.run(init)
    saver.restore(sess, path) #load the network
    
    tmp = tf.cast(final, tf.float32)
    acc = tf.nn.softmax(tmp) #tensor type
    
    print('prob. of this digit : ')
    print(sess.run(acc, feed_dict={x:data}))
 


mousebrush()
print drawn_data #784 np array
data = []
data.append(drawn_data)
customize_test(data)
