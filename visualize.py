import numpy as np
from PIL import Image
import os

npy_file1 = './submit_result/1110_1.npy'
npy_file2 = './submit_result/1110_2.npy'
npy_file3 = './submit_result/1110_3.npy'
npy_file4 = './submit_result/1110_4.npy'
npy_file5 = './submit_result/1110_5.npy'

arr1 = np.load(npy_file1)
arr2 = np.load(npy_file2)
arr3 = np.load(npy_file3)
arr4 = np.load(npy_file4)
arr5 = np.load(npy_file5)
print(sum(sum(arr1)))
print(sum(sum(arr2)))
print(sum(sum(arr3)))
print(sum(sum(arr4)))
print(sum(sum(arr5)))
arr1 = 50*arr1
arr2 = 50*arr2
arr3 = 50*arr3
arr4 = 50*arr4
arr5 = 50*arr5
img1 = Image.fromarray(arr1).convert("L")
img2 = Image.fromarray(arr2).convert("L")
img3 = Image.fromarray(arr3).convert("L")
img4 = Image.fromarray(arr4).convert("L")
img5 = Image.fromarray(arr5).convert("L")
img1.save("./test_pic/test1.png")
img2.save("./test_pic/test2.png")
img3.save("./test_pic/test3.png")
img4.save("./test_pic/test4.png")
img5.save("./test_pic/test5.png")
