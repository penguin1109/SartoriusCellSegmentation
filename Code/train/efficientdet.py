# the function for the first batch of training
def first_train(train_dl):
    """
    Input : 
    - train_dl : the dataloader for the training dataset
    GT :
    - Bounding boxes
    - Confidence Scores
    - Segmentation Mask
    PRED :
    - Bounding Boxes
    - Confidence Score
    - Instance Classes
    - Segmentation Mask
    """
    # use 'take(1)' to only pull one example from the dataset
    for _image_batch, _label_batch in train_dl.take(1):
        # get information for the ground truth data
        gt_mask = _label_batch["image_masks"][:,0]
        gt_boxes= _label_batch["groundtruth_data"][..., :4]
        gt_is_crowds = _label_batch["groundtruth_data"][..., 4]
        gt_areas = _label_batch["groundtruth_data"][..., 5]
        gt_classes = _label_batch["groundtruth_data"][..., 6]

        pred_classes, pred_boxes, pred_mask = model(_image_batch, training = False)
        pred_boxes, pred_scores, pred_classes, valid_len = postprocess.postprocess_global(config, pred_classes, pred_boxes)
        gt_instance_masks, pred_instance_masks = [], []
        for i in range(BATCH_SIZE):
            print("\n\n... ORIGINAL DISPLAY PLOT ...\n")
            _img, _mask = get_img_and_mask(**train_df.iloc[int(_label_batch["source_ids"][i])][["img_path", "annotation", "width", "height"]])
            plot_img_and_mask(_img, _mask)
            gt_instance_masks.append(_mask)

            print("\n... GROUND TRUTH PLOT ...\n")
            plot_gt(_image_batch[i], gt_classes[i], gt_boxes[i], gt_mask[i])

            print(f"\n... PREDICTION PLOT (NMS={'yes' if i<4 else 'no'}) ...\n")
            plot_pred(_image_batch[i], pred_boxes[i], pred_scores[i], pred_classes[i], pred_mask[i], iou_thresh=0.0 if i<4 else None)

            print(f"\n... GROUND TRUTH VS. PREDICTION PLOT (NMS={'yes' if i<4 else 'no'}) ...\n")
            plot_diff(_image_batch[i], gt_classes[i], gt_boxes[i], gt_mask[i], pred_boxes[i], pred_scores[i], pred_classes[i], pred_mask[i], iou_thresh=0.0 if i<4 else None)
        
            pred_instance_masks.append(get_pred_instance_mask(pred_boxes[i], pred_scores[i], pred_mask[i].numpy(), iou_thresh=0.0, conf_thresh=0.1))
        
            print("\n\n\n\n")
            print("-"*50)
            print("\n\n")
        
        print("\nBATCH_EVAL:\n")
        iou_map(gt_instance_masks, pred_instance_masks)

# function for the first batch of validation
def validate_first(val_dl):
    """
    Input :
    val_dl : the dataloader instance for the validation data
    Output :
    None
    """

# function for predicting the segmented cells for the test images  
def make_prediction(test_dl):
    """
    Input : 
    - test_dl (a dataloader for the efficientdet based on the test dataset)

    Output :
    - pred_instance_masks (list)
    : A list that includes the generated data containing the bbox and segmented cell class, etc
    """
    pred_instance_masks = []
    for _image_batch, _label_batch in test_dl:
        pred_classes, pred_boxes, pred_mask = model(_image_batch, training=False)
        pred_boxes, pred_scores, pred_classes, valid_len = postprocess.postprocess_global(config, pred_classes, pred_boxes)
        pred_instance_masks.append(get_pred_instance_mask(pred_boxes[0], pred_scores[0], pred_mask[0].numpy(), iou_thresh=0.0, conf_thresh=0.075))
    
    return pred_instance_masks

def make_submission(pred_instance_masks):
    """
    Input :
    - pred_instance_masks (list) 
    : A list that contains the mask and predicted boxes generated from the model using the test submission dataset(dataframe)

    Output :
    None. Only makes the csv file to be submitted and saves it in the given directory
    """
    submission_dfs = []
    for i, _id in enumerate(ss_df.id.to_list()):
        tmp_df = pd.DataFrame([_id,]*int(pred_instance_masks[i].max()), columns=["id"])
        tmp_df["predicted"] = [rle_encode(np.where(pred_instance_masks[i]==j, 1.0, 0.0)) for j in range(1, int(pred_instance_masks[i].max()+1))]
        submission_dfs.append(tmp_df)
    submission_df = pd.concat(submission_dfs).reset_index(drop=True)
    submission_df.to_csv("submission.csv", index=False)