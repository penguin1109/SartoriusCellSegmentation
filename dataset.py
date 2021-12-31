import torch, cv2
import torch.utils.data as data
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

def prepare_loaders(df, fold, batch_size):
    train_df = df[df.fold != fold].reset_index(drop = True)
    valid_df = df[df.fold == fold].reset_index(drop = True)

    train_dataset = BuildDataset(train_df,)
    valid_dataset = BuildDataset(valid_df, train = False)

    train_loader = DataLoader(train_dataset, batch_size = batch_size, num_workers = 1, shuffle = True, pin_memory = True)
    valid_loader = DataLoader(valid_dataset, batch_size = batch_size, num_workers = 1, shuffle = True, pin_memory = True)

    return train_loader, valid_loader

class BuildDataset(data.Dataset):
    def __init__(self, df, img_size = (512,512),train = True):
        self.df = df
        self.img_size = img_size
        self.img_paths = df['image_path'].values
        self.train = train
        self.transforms = get_transforms()
        try: # if there is no mask then only send images because it is the test data
            self.mask_paths = df['mask_path'].values
        except:
            self.mask_paths = None
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.mask_paths is not None:
            mask_path = self.mask_paths[index]
            mask = np.load(mask_path)
            if self.transforms:
                data = self.transforms(image = img, mask = mask)
                img = data['image']
                mask = data['mask']
            mask = np.expand_dims(mask, axis = 0)
            return img, mask
        else:
            if self.transforms:
                data = self.transforms(image = img)
                img = data['image']
            return img
    
    @property
    def get_transforms(self):
        data_transforms = {
            "train": A.Compose([
            A.Resize(self.img_size),
            A.Normalize(
                 mean=[0.485, 0.456, 0.406], 
                 std=[0.229, 0.224, 0.225], 
                 max_pixel_value=255.0, 
                 p=1.0,
             ),
            A.ColorJitter(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=90, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
       
            ToTensorV2()], p=1.0),
    
        "valid": A.Compose([
            A.Resize(self.img_size),
            A.Normalize(
                 mean=[0.485, 0.456, 0.406], 
                 std=[0.229, 0.224, 0.225], 
                 max_pixel_value=255.0, 
                 p=1.0
             ),
            ToTensorV2()], p=1.0)
        }

        if self.train:
            return data_transforms['train']
        else:
            return data_transforms['valid']
