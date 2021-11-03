import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import plotly.express as px

from IPython.display import display
from PIL import Image, ImageEnhance
import PIL
import cv2
import os

print("\n...SET DATA ROUTES...\n")
ROOT_DIR = '/content/drive/MyDrive/AIforMedicalDiagnosis/CELLKAGGLE'
DATA_DIR = '/content/drive/MyDrive/AIforMedicalDiagnosis/CELLKAGGLE/DATA'

LC_DIR = os.path.join(DATA_DIR, "LIVECell_dataset_2021") # local directory
LC_ANN_DIR = os.path.join(LC_DIR, "annotations") # annotation directory
LC_IMG_DIR = os.path.join(LC_DIR, "images") # image directory
TRAIN_DIR = os.path.join(DATA_DIR, "train") # train data directory
TEST_DIR = os.path.join(DATA_DIR, "test") # test data directory
SEMI_DIR = os.path.join(DATA_DIR, "train_semi_supervised") # semi supervised train data directory

print("\n... TRAIN DATAFRAME ...\n")

# FIX THE TRAIN DATAFRAME (GROUP THE RLEs TOGETHER)
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
train_df = pd.read_csv(TRAIN_CSV)
display(train_df)

print("\n... SS DATAFRAME ..\n")
SS_CSV = os.path.join(DATA_DIR, "sample_submission.csv")
ss_df = pd.read_csv(SS_CSV)
ss_df["img_path"] = ss_df["id"].apply(lambda x: os.path.join(TEST_DIR, x+".png")) # Capture Image Path As Well
display(ss_df)

CELL_TYPES = list(train_df.cell_type.unique())
FIRST_SHSY5Y_IDX = 0
FIRST_ASTRO_IDX  = 1
FIRST_CORT_IDX   = 2

# This is required for plotting so that the smaller distributions get plotted on top
ARB_SORT_MAP = {"astro":0, "shsy5y":1, "cort":2}

print("\n... CELL TYPES ..")
for x in CELL_TYPES: print(f"\t--> {x}")
    
print("\n\n... BASIC DATA SETUP FINISHING ...\n")

# 환경변수의 CUDA 정보를 할당해서 GPU를 tensorflow가 사용할 수 있도록 한다.
import os

os.enviorn["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.enviorn["CUDA_VISIBLE_DEVICES"] = "0" # number of gpu

# tensorflow의 경우에 모든 gpu의 메모리를 할당받은 후에 
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# 세션 사용시.
# session = tf.Session(config=config)
