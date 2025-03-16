import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#Load preprocess data
data = pd.read_csv("data/jump_data.csv")
x =  data.iloc[:, :-1].values #Features (MFCC)
y = data.iloc[:, -1].values #Labels ( 0 = no jump, 1 = jump)

#Split training and testing data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

#Create model for detecting jump

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(x_train.shape[1],)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid') #Output binary (0 or 1)
])

#Compile model
model.compile(optimize='adam', loss='binary_crossentropy', metrics=['accuracy'])

#Save model
model.save("models/jump_detection_model.h5")
print("Model save to models/jump_detection_model.h5")

