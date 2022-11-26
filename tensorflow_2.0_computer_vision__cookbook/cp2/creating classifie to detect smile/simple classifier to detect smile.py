import os 
import pathlib
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import Model
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing.image import *

def load_images_and_labels(image_paths):
    
    images=[]
    labels=[]
    
    for image_path in image_paths:
        image = load_img(inage_path, target_size=(32,32), color_mode='grayscale')
        image = img_to_array(image)
        
        label = image_path.split(os.path.sep)[-2]
        label = 'positive' in label
        label = float(label)
        
        images.append(image)
        labels.append(label)
        
    return np.array(images), np.array(labels)



def build_network():
    input_layer = Input(shape=(32,32,1))
    x = Conv2D(filters=20,
               kernel_size=(5, 5),
               padding='same',
               strides=(1, 1))(input_layer)
    x = ELU()(x)
    x = BatchNormalization()(x)
    X = MaxPooling2D(pool_size=(2,2),
                    strides=(2, 2))(x)
    x = Dropout(0.4)(x)
    
    x = Conv2D(filters=50,
               kernel_size=(5, 5),
               padding='same',
               strides=(1, 1))(x)
    x = ELU()(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2),
                     strides=(2, 2))(x)
    x = Dropout(0.4)(x)
    
    x = Flatten()(x)
    x = Dense(units=500)(x)
    x = ELU()(x)
    x = Dropout(0.4)(x)
    
    output = dense(1, activation='sigmoid')
    
    model = Model(inputs=input_shape, outputs=output)
    
    return model


files_pattern = (pathlib.Path.home() / '.keras' / 'datasets' /
                 'SMILEsmileD-master' / 'SMILEs' / '*' / '*' /
                 '*.jpg')
files_pattern = str(files_pattern)
dataset_paths = [*glob.glob(files_pattern)]



X, y = load_images_and_labels(dataset_paths)


X /= 255.0
total = len(y)
total_positive = np.sum(y)
total_negative = total - total_positive
print(f'Total images: {total}')
print(f'Smile images: {total_positive}')
print(f'Non-smile images: {total_negative}')


(X_train, X_test,
 y_train, y_test) = train_test_split(X, y,
                                     test_size=0.2,
                                     stratify=y,
                                     random_state=999)

(X_train, X_val, 
 y_train, y_val) = train_test_split(X_train,y_train, 
                                    test_size=0.2, 
                                    stratify=y_train, 
                                    random_state=999)



