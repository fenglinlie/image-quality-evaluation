import imageio
import numpy as np
import math

path_clear = './noise0%/'
path_noise = './data9%/'

psnr = 0
for i in range(160):
     image_name_clear = path_clear + str(i+1) + '.png'
     image_name_noise = path_noise + str(i+1) + '.png'
     img1 = imageio.imread(image_name_clear)
     img2 = imageio.imread(image_name_noise)
     mse = np.mean((img1/1.0 - img2/1.0)**2)
     if mse<1.0e-10:
        psnr += 100
     psnr += 10*math.log10(255.0**2/mse)
psnr = psnr/160
print(psnr)
