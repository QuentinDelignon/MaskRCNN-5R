import os 
import sys 
from PIL import Image
from tqdm import tqdm

sys.path.append('')

save_path = 'C:\\Users\\Quentin Delignon\\Documents\\Python\\PJE\\segmentation\\TestingImageDetection\\images\\'
list = os.listdir('C:\\Users\\Quentin Delignon\\Documents\\Python\\PJE\\segmentation\\TestingImageDetection\\Images1\\')
list2 = os.listdir('C:\\Users\\Quentin Delignon\\Documents\\Python\\PJE\\segmentation\\TestingImageDetection\\Images2\\')
list = ['C:\\Users\\Quentin Delignon\\Documents\\Python\\PJE\\segmentation\\TestingImageDetection\\Images1\\' + i for i in list]
list2 = ['C:\\Users\\Quentin Delignon\\Documents\\Python\\PJE\\segmentation\\TestingImageDetection\\Images2\\' + i for i in list2]
for item in list2:
	list.append(item)
for i in tqdm(range(len(list))):
	img = Image.open(list[i])
	img.save(save_path+'img_%d.jpg'%(i))
