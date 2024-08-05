#import os
#import pickle

#actors = os.listdir('data')
#filenames = []

#for actor in actors:
    #for file in os.listdir(os.path.join('data',actor)):
        #filenames.append(os.path.join('data',actor,file))

#pickle.dump(filenames,open('filenames.pkl','wb'))
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,Flatten,Dropout , BatchNormalization
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import pickle
from tqdm import tqdm

filenames = pickle.load(open('filenames.pkl','rb'))

base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze the base model

model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))  # Regularization to prevent overfitting
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))  # Regularization to prevent overfitting
model.add(Dense(128, activation='relu'))  # Output features of dimension 128


def feature_extractor(img_path,model):
    img = image.load_img(img_path,target_size=(224,224))
    img_array = image.img_to_array(img)
    expanded_img = np.expand_dims(img_array,axis=0)
    preprocessed_img = preprocess_input(expanded_img)

    result = model.predict(preprocessed_img).flatten()

    return result

features = []

for file in tqdm(filenames):
    features.append(feature_extractor(file,model))

pickle.dump(features,open('embedding.pkl','wb'))


