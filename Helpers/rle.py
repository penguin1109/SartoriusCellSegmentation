def rle_decode(mask_rle, shape, color = 1):
    """
    Args : 
    mask_rle(str) : run-length as string formated (start length)
    shape(tuple of ints) : (height, width) of array to return

    Returns : 
    Mask (np.array)
    - 1 : indicating mask
    - 0 : indicating background
    """
    # split the string by space, and convert into a integer array
    s = np.array(mask_rle.split(), dtype = int)
    # every even value is the start, every odd value is the run length
    starts = s[0::2] - 1
    lengths = s[1::2]
    ends = starts + lengths
    # the image is actually flattened since RLE is a 1D "run"
    if len(shape) == 3:
        h,w,d = shape
        img = np.zeros((h*w, d), dtype = np.float32)
    else:
        h,w = shape
        img = np.zeros((h*w,),dtype = np.float32)
    # the color is actually just any integer you want
    for lo, hi in zip(starts, ends):
        img[lo:hi] = color
    return img.reshape(shape)

def rle_encode(img):
    """
    Args :
    img(np.array) : 
    - 1 : indicating mask
    - 0 : indicating background
    
    Returns :
    run length as string formated
    """
    pixs = img.flatten()
    pixs = np.concatenate([[0], pixs, [0]])
    runs = np.where(pixs[1:] != pixs[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def tf_load_png(img_path):
    return tf.image.decode_png(tf.io.read_file(img_path), channels = 3)

def get_img_and_mask(img_path, annotation, width, height, mask_only=False, rle_fn = rle_decode):
    """"Capture the relevant image array as well as the image mask"""
    img_mask = np.zeros((height, width), dtype = np.uint8)
    for i, annot in enumerate(annotation):
        img_mask = np.where(rle_fn(annot, (height, width))!=0, i, img_mask)
    
    if (mask_only):
        return img_mask
    
    img = tf_load_png(img_path)[...,0]
    return img, img_mask

def plot_img_and_mask(img, mask, bboxes = None, invert_img = True, boost_contrast = True):
    """
    Function to take an image and the corresponding mask and plot

    Args : 
    img(np.array) : 1 channel numpy array that represents the image of cellular structures
    mask(np.array) : 1 channel numpy array that represents the instance masks (incrementing by one)
    bboxes(list of tuples) : consists of enclosing bboxes
    invert_img (bool) : whether or not to invert the base image
    boost_contrast(bool) : whether or not to boost the contrast of the base image

    Returns : 
    None : Plots the two arrays and overlays them to create a merged image
    """
    plt.figure(figsize = (20, 10))
    plt.subplot(1,3,1)
    _img = np.tile(np.expand_dims(img, axis = -1), 3)
    #convert white->black, black->white
    if (invert_img):_img = _img.max() - _img
    if (boost_contrast):_img = np.asarray(ImageEnhance.Contrast(Image.fromarray(_img)).enhance(16))
    if (bboxes):
        for i, bbox in enumerate(bboxes):
            mask = cv2.rectangle(mask, bbox[0], bbox[1], (i+1, 0, 0), thickness = 2)
        
    plt.imshow(_img)
    plt.axis(False)
    plt.title("Cell Image", fontweight = "bold")
    
    plt.subplot(1,3,2)
    _mask = np.zeros_like(_img)
    _mask[...,0] = mask
    plt.imshow(mask, cmap = "inferno")
    plt.axis(False)
    plt.title("Instance Segmentation on Mask", fontweight = "bold")

    merged = cv2.addWeighted(_img, 0.75, np.clip(_mask, 0, 1)*255, 0.25, 0.0)
    plt.subplot(1,3,3)
    plt.imshow(merged)
    plt.axis(False)
    plt.title("Cell Image with Instance Segmentation Mask Overlay", fontweight = "bold")

    plt.tight_layout()
    plt.show()