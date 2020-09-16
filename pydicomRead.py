import pydicom
filepath="./data/brats/BraTS19_2013_11_1/"
data = pydicom.dcmread(filepath+"BraTS19_2013_11_1_seg.dcm")
print(data.data_element)