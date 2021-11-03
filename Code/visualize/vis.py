from Helpers.rle import get_img_and_mask
import utils
import numpy as np
import matplotlib.pyplot as plt

def view_single_img(train_df, demo_idx = 11):
    img, mask = utils.get_img_and_mask(**train_df[["img_path", "annotation", "width", "height"]].iloc[demo_idx].to_dict())
    utils.plot_img_and_mask(img, mask)

    plt.figure(figsize = (20, min(80, mask.max()//2)))
    for i in range(1, mask.max()+1):
        plt.subplot(10,10,i)
        tl, br  = utils.get_contour_bbox(np.where(mask==i, 1, 0).astype(np.uint8))
        plt.imshow(np.asarray(ImageEnhance.Contrast(Image.fromarray(255-img.numpy())).enhance(16))[tl[1]:br[1], tl[0]:br[0]], cmap="magma")
        plt.axis(False)
        plt.title(f"{i}",fontweight = "bold")
        if i == 100:break

    plt.tight_layout()
    plt.show()

def pd_get_bboxes(row):
    """Get all bboxes for a given row/cell image"""
    mask = get_img_and_mask(row.img_path, row.annotation, row.height, mask_only=True)
    return [get_contour_bbox(np.where(mask==i, 1, 0).astype(np.uint8)) for i in range(1, mask.max()+1)]
    
def add_bbox_to_df(train_df):
    print("\n... CREATE FULL SCALE BBOXES ...\n")
    train_df["bboxes"] = train_df.parallel_apply(pd_get_bboxes, axis = 1)
    display(train_df.head())

    print("\n... CREATE SCALED DOWN (0-1) BBOXES ...\n")
    IMG_O_W, IMG_O_H = train_df.iloc[0].width, train_df.iloc[0].height
    train_df["scaled_bboxes"] = train_df.bboxes.progress_apply(lambda box_list: [((box[0][0]/IMG_O_W, box[0][1]/IMG_O_H), (box[1][0]/IMG_O_W,box[1][1]/IMG_O_H)) if box else None for box in box_list])

    # CORT
    img, msk = get_img_and_mask(**train_df[["img_path", "annotation", "width", "height"]].iloc[FIRST_SHSY5Y_IDX].to_dict())
    plot_img_and_mask(img, msk, bboxes=train_df.iloc[FIRST_SHSY5Y_IDX].bboxes)

    # ASTRO
    img, msk = get_img_and_mask(**train_df[["img_path", "annotation", "width", "height"]].iloc[FIRST_ASTRO_IDX].to_dict())
    plot_img_and_mask(img, msk, bboxes=train_df.iloc[FIRST_ASTRO_IDX].bboxes)

    # SHSY5Y
    img, msk = get_img_and_mask(**train_df[["img_path", "annotation", "width", "height"]].iloc[FIRST_CORT_IDX].to_dict())
    plot_img_and_mask(img, msk, bboxes=train_df.iloc[FIRST_CORT_IDX].bboxes)