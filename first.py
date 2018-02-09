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
          
          break
      break
      
  cv2.destroyAllWindows()
  
def customize_test(data):
  prob = 1
  path = '/mnt/disk2/saved_mnist/model2/model_new.ckpt'
  
  def model(inputs):
    num_layer1 = 100
    w1 = tf.Variable(tf.truncated_normal(shape=[784, num_layer1], stddev = 0.01))
    b1 = tf.Variable(tf.constant(1e-5, shape=[num_layer1]))
    l1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(inputs, w1), b1))
    
    
