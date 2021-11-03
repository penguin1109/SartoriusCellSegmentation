IMAGE_SHAPE = (train_df.iloc[0].height, train_df.iloc[0].width, 3)
INPUT_SHAPE = (640, 640, 3)
SEG_SHAPE = (INPUT_SHAPE[0]//4, INPUT_SHAPE[1]//4, 1)
MODEL_LEVEL = "d1" 
MODEL_NAME = f"efficientdet-{MODEL_LEVEL}"
BATCH_SIZE = 8
N_EVAL = 50
N_TRAIN = len(train_df)-N_EVAL
N_TEST = len(ss_df) # sample submission의 데이터 row개수 -> test에 사용될 제출 데이터의 개수
N_EXPOCH = 40
N_EX_PER_REC = 280
CLASS_LABELS = list(train_df.cell_type.unique())
N_CLASSES_OD = len(CLASS_LABELS)+1 # background + 3 cell types
N_CLASSES_SEG = 2 # background + cell
MAX_N_INSTANCES = int(100*np.ceil(train_df.bboxes.apply(len).max()/100))

DO_TRAIN = True
PRETRAINED_MODEL_DIR = '/content/drive/MyDrive/AIforMedicalDiagnosis/CELLKAGGLE/MODEL/efficientdet-d1.tar.gz'

print("\n ... HYPERPARAMETER CONSTANTS ...")
print(f"\t--> MODEL NAME         : {MODEL_NAME}")
print(f"\t--> BATCH SIZE         : {BATCH_SIZE}")
print(f"\t--> IMAGE SHAPE        : {IMAGE_SHAPE}")
print(f"\t--> INPUT SHAPE        : {INPUT_SHAPE}")
print(f"\t--> SEGMENTATION SHAPE : {SEG_SHAPE}")

config = hparams_config.get_efficientdet_config(MODEL_NAME)
KEY_CONFIGS = [
    "name", "image_size", "num_classes", "seg_num_classes", "heads", "train_file_pattern",
    "val_file_pattern", "model_name", "model_dir", "pretrained_ckpt", "batch_size", "eval_samples",
    "num_examples_per_epoch", "num_epochs", "steps_per_execution", "steps_per_epoch", 
    "profile", "val_json_file", "max_instances_per_image", "mixed_precision", 
    "learning_rate", "lr_warmup_init", "mean_rgb", "stddev_rgb","scale_range",
              ]

for k in config.keys():
    if k=="model_optimizations":
        continue
    elif k=="nms_configs":
        for _k, _v in dict(config[k]).items():
            print(f"PARAMETER: {'     ' if _k not in KEY_CONFIGS else ' *** '}nms_config_{_k: <16}  ---->    VALUE:  {_v}")
        
    else:
        print(f"PARAMETER: {'     ' if k not in KEY_CONFIGS else ' *** '}{k: <27}  ---->    VALUE:  {config[k]}")