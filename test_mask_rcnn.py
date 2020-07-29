from torchvision.transforms import functional as F
import torch
from PIL import Image
import warnings
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')


threshold = 0.5
model = torch.load('C:\\Users\\Quentin Delignon\\Documents\\Python\\PJE\\segmentation\\TestingImageDetection\\MaskRCNN.pth')
img = Image.open('C:\\Users\\Quentin Delignon\\Documents\\Python\\PJE\\segmentation\\TestingImageDetection\\images\\img_0.jpg').convert('RGB')
img = F.to_tensor(img)
pred = model([img])
print(pred[0].keys())
#print(pred[0])
pred_score = list(pred[0]['scores'].detach().numpy())
threshold = np.median(pred_score)
pred_t = [pred_score.index(x) for x in pred_score if x>threshold][-1]
masks = (pred[0]['masks']>0.5).squeeze().detach().cpu().numpy()
pred_class = [str(i) for i in list(pred[0]['labels'].numpy())]
pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())]
masks = masks[7:]
pred_boxes = pred_boxes[7:]
pred_class = pred_class[7:]
plt.imshow(img.permute(1,2,0).numpy())
plt.show()
for m in masks:
    if np.zeros_like(m) != m:
        print('this one is not void !')
    plt.imshow(m)
    plt.show()
