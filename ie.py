import tensorflow as tf


def IE(image):

    image = image*255  # float32类型的图像的数值范围在0.0 to 1.0 乘以255转化成uint8类型范围
    n = 255
    value_range = [0, 256]
    hist = tf.histogram_fixed_width(image, value_range, nbins=n, dtype=tf.int32)

    p1 = hist / tf.reduce_sum(hist)

    loss = -tf.reduce_sum(p1 * tf.log(p1 + 1e-8))
    loss = tf.to_float(loss)

    return loss


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

path = './DESN-SL-smoothing-15/'

ie = 0
for i in range(160):
    image_name = path + str(i+1) 
    
    img = load_image(image_name, 256)
    
    ie += IE(img)

ie = ie/160

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
print(ie.eval())
    













