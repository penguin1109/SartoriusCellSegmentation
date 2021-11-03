# utility for efficient detection model and constants
IMAGE_SHAPE = (train_df.iloc[0].height, train_df.iloc[0].width, 3)
INPUT_SHAPE = (640, 640, 3)
SEG_SHAPE = (INPUT_SHAPE[0]//4, INPUT_SHAPE[1]//4, 1)
MODEL_LEVEL = "d1" 
MODEL_NAME = f"efficientdet-{MODEL_LEVEL}"
BATCH_SIZE = 8
N_EVAL = 50
N_TRAIN = len(train_df)-N_EVAL
N_TEST = len(ss_df) # sample submission의 데이터 row개수 -> test에 사용될 제출 데이터의 개수
N_EPOCH = 40
N_EX_PER_REC = 280
CLASS_LABELS = list(train_df.cell_type.unique())
N_CLASSES_OD = len(CLASS_LABELS)+1 # background + 3 cell types
N_CLASSES_SEG = 2 # background + cell
MAX_N_INSTANCES = int(100*np.ceil(train_df.bboxes.apply(len).max()/100))

DO_TRAIN = True
PRETRAINED_MODEL_DIR = '/content/drive/MyDrive/AIforMedicalDiagnosis/CELLKAGGLE/MODEL/efficientdet-d1.tar.gz'

config = hparams_config.get_efficientdet_config(MODEL_NAME)
KEY_CONFIGS = [
    "name", "image_size", "num_classes", "seg_num_classes", "heads", "train_file_pattern",
    "val_file_pattern", "model_name", "model_dir", "pretrained_ckpt", "batch_size", "eval_samples",
    "num_examples_per_epoch", "num_epochs", "steps_per_execution", "steps_per_epoch", 
    "profile", "val_json_file", "max_instances_per_image", "mixed_precision", 
    "learning_rate", "lr_warmup_init", "mean_rgb", "stddev_rgb","scale_range",
              ]


DO_ADV_PROP=True
MODEL_DIR = f"/content/drive/MyDrive/AIforMedicalDiagnosis/CELLKAGGLE/MODEL/{MODEL_NAME}-finetune"

"""
if TPU:
    TFRECORD_DIR = os.path.join(KaggleDatasets().get_gcs_path('effdet-d5-dataset-sartorius'), "tfrecords")
else:
    TFRECORD_DIR = "/kaggle/working/tfrecords"
"""

TFRECORD_DIR = '/content/drive/MyDrive/AIforMedicalDiagnosis/CELLKAGGLE/tensorflow-model-record'
os.makedirs(MODEL_DIR, exist_ok=True)
config = hparams_config.get_efficientdet_config(MODEL_NAME)
overrides = dict(
    train_file_pattern=os.path.join(TFRECORD_DIR, "train", "*.tfrec"),
    val_file_pattern=os.path.join(TFRECORD_DIR, "val", "*.tfrec"),
    test_file_pattern=os.path.join(TFRECORD_DIR, "test", "*.tfrec"),
    model_name=MODEL_NAME,
    model_dir=MODEL_DIR,
    pretrained_ckpt=MODEL_NAME,
    batch_size=BATCH_SIZE,
    eval_samples=N_EVAL,
    num_examples_per_epoch=N_TRAIN,
    num_epochs=N_EPOCH,
    steps_per_execution=1,
    steps_per_epoch=N_TRAIN//BATCH_SIZE,
    profile=None, val_json_file=None,
    heads = ['object_detection', 'segmentation'],
    image_size = INPUT_SHAPE[:-1],
    num_classes = N_CLASSES_OD,
    seg_num_classes = N_CLASSES_SEG,
    max_instances_per_image = MAX_N_INSTANCES,
    input_rand_hflip=False, jitter_min=0.99, jitter_max=1.01,
    skip_crowd_during_training=False,
    )
config.override(overrides, True)
config.nms_configs.max_output_size = MAX_N_INSTANCES

# Change how input preprocessing is done
if DO_ADV_PROP:
    config.override(dict(mean_rgb=0.0, stddev_rgb=1.0, scale_range=True), True)


tf.keras.backend.clear_session()

model = efficientdet_keras.EfficientDetModel(config=config)
model.build((1,*INPUT_SHAPE))
model.summary()

print("\n... MODEL PREDICTIONS ...\n")
preds = model.predict(np.zeros((1,*INPUT_SHAPE)))
for i, name in enumerate(["bboxes", "confidences", "classes", "valid_len", "segmentation map"]):
    print(name)
    print(preds[i].shape)
    try:
        if preds[i].shape[-2]==64:
            print(preds[i][0, 0, 0, :5])
        else:
            print(preds[i][0, :5])
        
    except:
        print(preds[i][0])
    print()

# create model and load pretrained weights
if DO_TRAIN:
    # NOT evaluation mode TRAINING MODE
    if not os.path.isdir(MODEL_NAME):
        if DO_ADV_PROP:
            !wget https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/advprop/{MODEL_NAME}.tar.gz
        else:
            !wget https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco2/{MODEL_NAME}.tar.gz
        !tar -zxf {MODEL_NAME}.tar.gz
        !rm -rf {MODEL_NAME}.tar.gz

with strategy.scope():
    model = train_lib.EfficientDetNetTrain(config = config)
    model = setup_model(model, config)

    if DO_TRAIN:
        util_keras.restore_ckpt(
            model = model,
            ckpt_path_or_file = tf.train.latest_checkpoint(MODEL_NAME),
            ema_decay = config.moving_average_decay,
            exclude_layers = ['class_net']
        )
        ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
            os.path.join(MODEL_DIR, 'ckpt-{epoch:d}'),
            verbose = 1, save_freq="epoch", save_weights_only = True
        )
    else:
        model.load_weights(os.path.join(PRETRAINED_MODEL_DIR, "ckpt"))

model.summary()

# train the model
if DO_TRAIN:
    history = model.fit(
        train_df,
        epochs = config.num_epochs,
        steps_per_epoch = config.steps_per_epoch,
        callbacks = [ckpt_cb],
        validation_data = val_dl,
        validation_steps = N_EVAL//BATCH_SIZE
    )

else:
    print(model.evaluate(train_dl, steps = config.steps_per_epoch))
    print(model.evaluate(val_dl, steps = N_EVAL//BATCH_SIZE))