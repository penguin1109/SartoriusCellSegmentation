"""
INPUT : 
Raw Image (256, 256, 3)

OUTPUT : 
- Bounding Boxes
- Instance Classes
- Segmented Image (64, 64, 3)
"""
import Models.settings
import numpy as np
import cv2

# create the tfrecord datasets
def create_id_to_iloc_map(df):
    """
    create mapping to allow for numeric file names
    -> index in original df -> id
    """
    return {v:k for k,v in df.id_to_dict().items()}

TRAIN_ID_2_ILOC = create_id_to_iloc_map(train_df) # train csv
TEST_ID_2_ILOC = create_id_to_iloc_map(ss_df) # submision csv

def tf_load_image(path, resize_to = INPUT_SHAPE):
    # 학습이나 예측을 하기 전에 모델에 넣어주고자 하는 이미지의 형태로 돌려주는 함수
    """Load an image with the correct shape using tensorflow
    
    Args : 
    - path (tf.strings) : path to the correct shape using only tensorflow
    - resize_to (tuple) : the correct size to reshape the input image
    
    Returns :
    - 2 Channel tf.constant image ready for training and inference"""
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_png(img_bytes, channel = resize_to[-1])
    img = tf.image.resize(img, resize_to[:-1])
    img = tf.cast(img, tf.uint8)

    return img

def load_npz(path, resize_to=SEG_SHAPE, to_binary=True):
    np_arr = np.load(path)["arr_0"]
    if to_binary:
        return np.where(cv2.resize(np_arr, resize_to[:-1])>0, 1, 0).reshape(resize_to).astype(np.uint8)
    else:
        return cv2.resize(np_arr, resize_to[:-1]).reshape(resize_to).astype(np.int32)

