from Helpers.rle import rle_encode

ss_df["predicted"] = rle_encode(np.clip(msk, 0, 1))
ss_df = ss_df[["id", "predicted"]]
ss_df.to_csv("submission.csv", index=False)