import tensorflowjs as tfjs
import os
import tensorflow as tf

#Model path
MODEL_PATH = "models/jump_model.h5"
OUTPUT_PATH = "models/model_tfjs"

#Convert to tensorflow.js
tfjs.converters.save.keras.model(tf.keras.models.load_model(MODEL_PATH), OUTPUT_PATH)
print(f"Model converted to tensorflow.js in {OUTPUT_PATH}")