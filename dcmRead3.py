import SimpleITK as sitk
import numpy as np
from matplotlib import pyplot

file = sitk.ReadImage('./data/brats/BraTS19_2013_11_1/BraTS19_2013_11_1_flair.dcm')
print(file.GetSize())
print(file.GetOrigin()) # 坐标原点
print(file.GetSpacing()) # 像素间距
print(file.GetDirection()) # 方向
pixel_array = sitk.GetArrayFromImage(file) # 像素矩阵
# print(pixel_array.shape) # 打印矩阵维度
image_array = np.squeeze(pixel_array)
print(image_array.shape)
image_array = image_array.transpose(1,2,0)
print(image_array.shape)
# print(image_array.)

# pyplot.imshow(image_array)
# pyplot.show()