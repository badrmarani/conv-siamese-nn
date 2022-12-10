from model import ConvSiameseNet
import torch


from PIL import Image
import numpy as np

im = Image.open("test_im_gucci.png").convert("RGB")
im = np.asarray(im)
print(im.shape)
