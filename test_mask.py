from PIL import Image,ImageDraw
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
from PIL import Image,ImageDraw
from torchvision.transforms import functional as F
import torch.nn.functional as f
import warnings
import cv2
from tqdm import tqdm
warnings.simplefilter('ignore')

colormap = [[255,0,0],
            [255,255,0],
            [255,255,255],
            [0,0,0],
            [0,255,0],
            [0,0,255],
            [255,0,255],
            [0,255,255],
            [100,0,100],
            [100,0,0]]
import matplotlib.pyplot as plt
def make_masks(m,draw,labels):
    arr = np.array(draw)
    for i in range(len(m)):
        clr = colormap[int(labels[i])]
        mask = m[i].squeeze()
        mask = f.threshold(mask,0.5,1)
        mask[mask!=1] = torch.zeros_like(mask[mask!=1])
        mask = mask.detach().cpu().numpy()
        arr[mask == 0 ] = clr
    return Image.fromarray(arr,mode='RGB')

def make_boxes(b,draw,labels):
    for i in range(len(b)):
        clr = tuple(colormap[int(labels[i])])
        draw.rectangle(b[i].detach().cpu().numpy(),outline=clr)
    return


def getImage(img):
    device = torch.device('cpu')
    path = 'C:\\Users\\Quentin Delignon\\Documents\\Python\\PJE\\segmentation\\TestingImageDetection\\MASK_RCNN.pth'
    model = torch.load(path,map_location=device)
    model.eval()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    #img = Image.open('C:\\Users\\Quentin Delignon\\Documents\\Python\\PJE\\segmentation\\TestingImageDetection\\images\\img_%d.jpg'%(i)).convert("RGB")
    image = F.to_tensor(img).unsqueeze(0).to(device).detach()
    dic = model(image)[0]
    scores = dic['scores']
    labels = []
    masks = []
    boxes = []
    for i in range(scores.shape[0]):
        if scores[i] > 0.5:
            labels.append(dic['labels'][i])
            masks.append(dic['masks'][i])
            boxes.append(dic['boxes'][i])
    draw = ImageDraw.Draw(img)
    boxes = make_boxes(boxes,draw,labels)
    del draw
    result = make_masks(masks,img,labels)
    result = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
    return result
    #result.save('C:\\Users\\Quentin Delignon\\Documents\\Python\\PJE\\segmentation\\TestingImageDetection\\result_%d.png'%(i))
vid_path = 'C:\\Users\\Quentin Delignon\\Documents\\Python\\PJE\\segmentation\\Vidéos2\\test_matthieu.mp4'
cap = cv2.VideoCapture(vid_path)
active,frame = cap.read()
active,frame = cap.read()
frames_mod = []
frames_mod.append(frame)
while active:
    active,frame = cap.read()
    try:
        frames_mod.append(frame)
    except:
        pass
cap.release()
frames_mod = frames_mod[:-1]
height, width, layers = frames_mod[-5].shape
size = (width,height)
out = cv2.VideoWriter("output_matthieu.avi", cv2.VideoWriter_fourcc(*"XVID"), 15,size)
for i in tqdm(range(len(frames_mod))):
    im = getImage(frames_mod[i])
    out.write(im)
out.release()
