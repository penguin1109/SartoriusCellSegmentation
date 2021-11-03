import tensorflow as tf

print(f"\n... ACCELERATOR SETUP STARTING ...\n")
# Detect the Hardware, return the appropriate distribution strategy

try:
    # detect for TPU
    TPU = tf.distribute.cluster_resolver.TPUClusterResolver()
except ValueError:
    TPU = None

if TPU:
    print(f"\n... RUNNING ON TPU-{TPU.master()}...")
    tf.config.experimental_connect_to_cluster(TPU)
    tf.tpu.experimental.initialize_tpu_system(TPU)
    strategy = tf.distriute.experimental.TPUStrrategy(TPU)
else:
    print(f"\n... RUNNING ON GPU ...")
    # use the default distribution strategy in Tensorflow
    strategy = tf.distribute.get_strategy()
    
print(f"\n... ACCELERATOR SETUP ENDED ...")
