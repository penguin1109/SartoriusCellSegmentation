import cupy as cp
import skimage.morphology
import numpy as np
import torch

def ins2rle(ins):
    '''
    img : numpy array, 1(mask), 0(background)
    returns a run length encoded string format
    '''
    ins = cp.array(ins)
    pixels = ins.flatten()
    pad = cp.array([0])
    pixels = cp.concatenate([pad, pixels, pad])
    runs = cp.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def mask2rle(mask, cutoff = 0.5, min_object_size = 1.0):
    '''
    returns a runlength encoding of the mask
    the mask is not a binary mask, but the mask that the model gives as an output that passes the nn.Sigmoid function
    '''
    # segment image and label different objects
    lab_mask = skimage.morphology.label(mask > cutoff)

    # keep only the objects that are large enough to be considered as an instance cell
    (mask_labels, mask_sizes) = np.unique(lab_mask, return_counts = True)
    if (mask_sizes  < min_object_size).any():
        mask_labels = mask_labels[mask_sizes < min_object_size]
        for n in mask_labels:
            lab_mask[lab_mask == n] == 0 # label the 'too small' instances as the background
        lab_mask = skimage.morphology.label(lab_mask > cutoff)

    # loop over each object excluding the background labeled by 0
    for i in range(1, lab_mask.max() + 1):
        yield ins2rle(lab_mask == 1) # rle encode the instances segmented from the background, labeled as 0

def aug(img, axis = 0):
    if axis == 1:
        return torch.flip(img, dims = (1,))
    elif axis == 2:
        return torch.filp(img, dim = (2,))
    elif axis == 3:
        return torch.flip(img, dims = (1,2))
    elif axis == 4:
        return torch.rot90(img, k = 1, dims = (1,2))
    elif axis == 5:
        return torch.rot90(img, k = 1, dims = (2,1))
    else:
        return img

def reverse_aug(img, axis=0):
    if axis == 1:
        return torch.flip(img,dims=(1,))
    elif axis == 2:
        return torch.flip(img,dims=(2,))
    elif axis == 3:
        return torch.flip(img,dims=(1,2))
    elif axis == 4:
        return torch.rot90(img, k=1, dims=(2,1))
    elif axis == 5:
        return torch.rot90(img, k=1, dims=(1,2))
    else:
        return img

def get_aug_img(img, ttas = CFG.ttas):
    