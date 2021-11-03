def agg_under_training(train_df):
    # Aggregate under training 
    train_df["img_path"] = train_df["id"].apply(lambda x: os.path.join(TRAIN_DIR, x+".png")) # Capture Image Path As Well
    tmp_df = train_df.drop_duplicates(subset=["id", "img_path"]).reset_index(drop=True)
    tmp_df["annotation"] = train_df.groupby("id")["annotation"].agg(list).reset_index(drop=True)
    train_df = tmp_df.copy()
    #train_df["seg_path"] = train_df.id.apply(lambda x: os.path.join(SEG_DIR, f"{x}.npz"))
    display(train_df)

def visualize_train_data():
    for i in range(2, 70, 8):
        print(f"\n\n\n... RELEVANT DATAFRAME ROW - INDEX={i} ...\n")
        display(train_df.iloc[i:i+1])
        img, mask = get_img_and_mask(**train_df[["img_path", "annotation", "width", "height"]].iloc[i].to_dict())
        plot_img_and_mask(img, mask)

