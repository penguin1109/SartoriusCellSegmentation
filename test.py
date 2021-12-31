import torch.nn as nn
import torch, cv2
from utils import rle_encode, rle_decode, get_img
import numpy as np
from model import CellUNetPP
import pandas as pd

def get_masks(file_name, net):
  img = get_img(file_name, image_enhance = False)
  x_, y_ = [0, 248], [0,232,448]
  predict = torch.zeros_like(img)
  for i in x_:
    for j in y_:
      croped_img = img[:,:,i:i+256,j:j+256]
      croped_img.cuda(
      predict[:,:,i:i+256, j:j+256] += net(croped_img)
  predict = torch.softmax(predict, dim = 1)
  predict = torch.argmax(predict, dim = 1, keepdim = True)
  predict = totch.where(predict !=0,1,predict).cpu().numpy()
  res = []
  used = np.zeros(img.shape[:2], dtype = int)
  for mask in predict:
    mask = mask * (1 - used)
    used += mask
    res.append(rle_encode(mask))
  return res

def show():
  _, axs = plt.subplots(1,2, figsize=(40,15))
  axs[1].imshow(cv2.imread(str(test_names[i])))
  for enc in encoded_masks:
    dec = rle_decode(enc)
    axs[0].imshow(np.ma.masked_where(dec==0, dec))

def main():
  dataDir=Path('../input/sartorius-cell-instance-segmentation')
  net = CellUNetPP()
  ids, masks=[],[]
  test_names = (dataDir/'test').ls()
  for fn in test_names:
    encoded_masks = get_masks(file_name = fn,net)
    for enc in encoded_masks:
        ids.append(fn.stem)
        masks.append(enc)
  
  pd.DataFrame({'id':ids, 'predicted':masks}).to_csv('submission.csv', index=False)
  pd.read_csv('submission.csv').head()