# UPDATE THE TRAIN DATAFRAME

# Aggregate under training 
from load import CELL_TYPES


train_df["img_path"] = train_df["id"].apply(lambda x: os.path.join(TRAIN_DIR, x+".png")) # Capture Image Path As Well
tmp_df = train_df.drop_duplicates(subset=["id", "img_path"]).reset_index(drop=True)
tmp_df["annotation"] = train_df.groupby("id")["annotation"].agg(list).reset_index(drop=True)
train_df = tmp_df.copy()
train_df["seg_path"] = train_df.id.apply(lambda x: os.path.join(SEG_DIR, f"{x}.npz"))
display(train_df)


# VISUALIZE THE TRAIN DATA
for i in range(2, 70, 8):
    print(f"\n\n\n\n... RELEVANT DATAFRAME ROW - INDEX={i} ...\n")
    display(train_df.iloc[i:i+1])
    img, msk = get_img_and_mask(**train_df[["img_path", "annotation", "width", "height"]].iloc[i].to_dict())
    plot_img_and_mask(img, msk)

# INVESTIGATE THE TRAIN DATAFRAME
print("\n\n... WIDTH VALUE COUNTS ...")
# 원본 이미지의 가로 길이
for k, v in train_df.width.value_counts().items():
    print(f"\t--> There are {v} images with WIDTH={k}")

print("\n\n... HEIGHT VALUE COUNTS ...")
# 원본 이미지의 세로 길이
for k, v in train_df.height.value_counts().items():
    print(f"\t--> There are {v} images with HEIGHT={k}")

print("\n\n... AREA COUNTS ...")
# 원본 이미지의 넓이
for k, v in (train_df.height * train_df.width).value_counts().items():
    print(f"\t-->There are {v} images with AREA={k}")

print("\n\n... NOTE: ALL THE IMAGES ARE THE SAME SIZE ...\n")

print("\n\n... PLATE TIME VALUE COUNTS ...")
for k, v in train_df.plate_time.value_counts().items():
    print(f"\t--> There are {v} images with SAMPLE_DATe={k}")
fig = px.histogram(train_df, train_df.sample_date.apply(lambda x:x.replace("-", "_")), color = "cell_type", title = "<b>Sample Date Value Histogram</b>")
fig.show()

print("\n\n...SAMPLE DATE VALUE COUNTS...")
for k, v in train_df.sample_date.value_counts().items():
    print(f"\t-->There are {v} images with SAMPLE_DATE={k}")
fig = px.histogram(train_df, "elapsed_timedelta", color="cell_type", title="<b>Elapsed Time Delta</b>")

for ct in CELL_TYPES:
    print(f"\n\n ...SHOWING THREE EXAMPLES OF CELL TYPE{ct.upper()} ..\n")
    for i in range(3):
        img, mask = get_img_and_mask(**train_df[train_df.cell_type==ct][["img_path", "annotation", "width", "height"]].sample(3).reset_index(drop=True).iloc[i].to_dict())
        plot_img_and_mask(img, mask)






