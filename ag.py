import tensorflow as tf
import numpy as np


def average_gradient(image):    
   

    a = tf.image.sobel_edges(image)     
    wx = a[:, :, :, :, 0]              
    wy = a[:, :, :, :, 1]             

    wx = tf.reshape(wx, [wx.shape[1].value, wx.shape[2].value, 1])
    wy = tf.reshape(wy, [wy.shape[1].value, wy.shape[2].value, 1])
  
    ag = tf.reduce_sum(tf.sqrt(tf.square(wx) + tf.square(wy) + 1e-8)) / (256*256*1)

    return ag


def load_image(filename, dim):
    with open(filename, 'rb') as f:
        raw_image = tf.image.decode_png(f.read())
    

    converted = tf.image.convert_image_dtype(
        raw_image,
        tf.float32,
        saturate = True
    )
  
    resized = tf.image.resize_images( 
        images = converted,
        size = [dim, dim]
    )

    resized.set_shape((dim, dim, 1))
    
    resized = tf.expand_dims(resized, 0)

    return resized


path = './DESN-SL-smoothing-35/'

ag = 0 

for i in range(160):
    image_name = path + str(i+1)
    img = load_image(image_name, 256)
    ag += average_gradient(img)

ag = ag/160


sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
print(ag.eval())



