# using tensorflow package
import tensorflow as tf
import numpy as np
import os


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


path_clear = './noise0%' 
path_noise = './data3%' 

ssim = 0

'''
image_name_clear = path_clear + str(1)
image_name_noise = path_noise + str(1)

img_clear = load_image(image_name_clear, 256)
img_noise = load_image(image_name_noise, 256)

ssim = tf.image.ssim(img_clear, img_noise, 1)
'''

for i in range(160):
    image_name_clear = path_clear + os.sep + str(i+1) + '.png'
    image_name_noise = path_noise + os.sep + str(i+1) + '.png'
    
    img_clear = load_image(image_name_clear, 256)
    img_noise = load_image(image_name_noise, 256)
    
    ssim += tf.image.ssim(img_clear, img_noise, 1)
     
ssim = ssim/160

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
print(ssim.eval())
