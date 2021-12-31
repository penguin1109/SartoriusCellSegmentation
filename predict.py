import torch.nn as nn
import torch, cv2, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
MODEL_DIR ='../input/sartorius-cell-instance-segmentation/unetpp/05.pth'
def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    
    return ' '.join(str(x) for x in runs)

def get_img(img_dir,axis = 1, image_enhance = True, enhance_rate = 17):
  img = cv2.imread(img_dir)
  img = img[...,0]
  
  if image_enhance:
    _img = np.tile(np.expand_dims(img, axis = -1),3)
    _img = _img.max() - _img
    _img = np.asarray(ImageEnhance.Contrast(Image.fromarray(_img)).enhance(enhance_rate))
  else:
    img = img / img.max()
    if axis == 3:
        _img = np.tile(np.expand_dims(img, axis = -1),3)
    else:
        _img = np.expand_dims(img, axis = -1)
    _img = _img.max() - _img

  return _img
def get_masks(file_name, net):
  img = get_img(file_name, axis = 1,image_enhance = False).astype('float32')
  #print(img.shape)
  if (img.shape[-1] == 1):
    img = np.transpose(img, (2,0,1))
    img = np.expand_dims(img, axis = 0)
  x_, y_ = [0, 132,264], [0,224,448]
  predict = torch.zeros((1,4,img.shape[2], img.shape[-1])).cuda()
  for i in x_:
    for j in y_:
      croped_img = img[:,:,i:i+256,j:j+256]
      
      croped_img_ = torch.from_numpy(croped_img).cuda()
      prediction = net(croped_img_)
      #print(prediction.shape)
      predict[:,:,i:i+256, j:j+256] += prediction.detach()
  predict = torch.softmax(predict, dim = 1)
  predict = torch.argmax(predict, dim = 1, keepdim = True)
  predict = torch.where(predict !=0,1,predict)
  return predict

def show():
  _, axs = plt.subplots(1,2, figsize=(40,15))
  axs[1].imshow(cv2.imread(str(test_names[i])))
  for enc in encoded_masks:
    dec = rle_decode(enc)
    axs[0].imshow(np.ma.masked_where(dec==0, dec))
    
def postprocess(mask, min_size=80, shape=(520, 704,)):
    
    #print(mask.shape)
    num_component, component = cv2.connectedComponents(mask)
    predictions = []
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            a_prediction = np.zeros(shape, np.float32)
            a_prediction[p] = 1
            predictions.append(a_prediction)
    return predictions

def remove_isolated_points_from_rle(strin):
    t2 = strin.split(" ")
    a = []
    for i in range(0, len(t2), 2):
        if t2[i+1]!="1":
            a.append(t2[i])
            a.append(t2[i+1])
    return ' '.join(a)

def main():
  #dataDir=Path('../input/sartorius-cell-instance-segmentation')
  net = CellUNetPP(num_classes = 4)
  ckpt_path = '../input/unetpp/05.pth'
  net.load_state_dict(torch.load(ckpt_path))
  net.eval()
  net.cuda()
  sample_submission = pd.read_csv('../input/sartorius-cell-instance-segmentation/sample_submission.csv')
  test_ids = sample_submission['id'].unique().tolist()
  
  ids, masks=[],[]
  TEST_PATH = '../input/sartorius-cell-instance-segmentation/test/'
  for id_ in test_ids:
    fn = TEST_PATH + id_+'.png'
    masks.append(get_masks(fn,net))
  predicted_nucleus = []
  test_nucleus_image_id = []

  for index, s in enumerate(masks):
    s = s.cpu().detach().numpy()[0,0,:,:].astype(np.uint8)
    s = cv2.resize(s, (704,520,))
    nucleus = postprocess(s)
    for nucl in nucleus:
        predicted_nucleus.append(nucl)
        test_nucleus_image_id.append(test_ids[index])
  predicted2 = [rle_encode(test_mask2) for test_mask2 in predicted_nucleus]
  print(predicted2[0])
  predicted_filt = [remove_isolated_points_from_rle(s) for s in predicted2]
  print(predicted_filt[0])
  pd.DataFrame({'id':test_nucleus_image_id, 'predicted':predicted_filt}).to_csv('submission.csv', index=False)
  pd.read_csv('submission.csv').head()


if __name__ == '__main__':
  main()