import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
image = sitk.ReadImage(r"D:\Python\PyCharm\100GTR\BraTS19_2013_11_1\BraTS19_2013_11_1_flair.dcm")
image_array = np.squeeze(sitk.GetArrayFromImage(image))
plt.imshow(image_array)
plt.show()
# dcm_dir = "./data/brats/BraTS19_2013_11_1/"


