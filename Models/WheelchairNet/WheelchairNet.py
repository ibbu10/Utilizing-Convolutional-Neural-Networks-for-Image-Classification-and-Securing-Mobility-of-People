
from keras.models import Sequential 
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
import keras
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt





classifier = Sequential()
    
        # Step1 Convolution
    
classifier.add(Convolution2D(32,3,3,input_shape=(128,128,3),activation = 'relu'))
    
        # Step 2 Pooling 
    
classifier.add(MaxPooling2D(pool_size=(2,2)))
    
        # Second Convolutional layer 
classifier.add(Convolution2D(32,3,3,activation = 'relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
        
        # third Convolutional layer 
classifier.add(Convolution2D(32,3,3,activation = 'relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

   # fourth Convolutional layer 
#classifier.add(Convolution2D(32,3,3,activation = 'relu'))
#classifier.add(MaxPooling2D(pool_size=(2,2)))
        
        # Step 3 Flattening 
        
classifier.add(Flatten())
    
        #step 4 Full Connection 
    
classifier.add(Dense(output_dim = 128 , activation = 'relu'))
classifier.add(Dropout(0.5))
   

classifier.add(Dense(output_dim = 1 , activation = 'sigmoid'))

# for custom metrics
"""import keras.backend as K

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

def false_rates(y_true, y_pred):
    false_neg = ...
    false_pos = ...
    return {
        'false_neg': false_neg,
        'false_pos': false_pos,
    }

classifier.compile(optimizer='Adadelta',
              loss='binary_crossentropy',
              metrics=['accuracy', mean_pred, false_rates])
"""

import keras.backend as K

# get F1 Score 
def f1_score(y_true, y_pred):
   
    tp_3d = K.concatenate(
        [
            K.cast(y_true, 'bool'),
            K.cast(K.round(y_pred), 'bool'),
            K.cast(K.ones_like(y_pred), 'bool')
        ], axis=1
    )

    fp_3d = K.concatenate(
        [
            K.cast(K.abs(y_true - K.ones_like(y_true)), 'bool'),
            K.cast(K.round(y_pred), 'bool'),
            K.cast(K.ones_like(y_pred), 'bool')
        ], axis=1
    )

    fn_3d = K.concatenate(
        [
            K.cast(y_true, 'bool'),
            K.cast(K.abs(K.round(y_pred) - K.ones_like(y_pred)), 'bool'),
            K.cast(K.ones_like(y_pred), 'bool')
        ], axis=1
    )

    tp = K.sum(K.cast(K.all(tp_3d, axis=1), 'int32'))
    fp = K.sum(K.cast(K.all(fp_3d, axis=1), 'int32'))
    fn = K.sum(K.cast(K.all(fn_3d, axis=1), 'int32'))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * ((precision * recall) / (precision + recall))

#Get Precision

def Precision(y_true, y_pred):
   
    tp_3d = K.concatenate(
        [
            K.cast(y_true, 'bool'),
            K.cast(K.round(y_pred), 'bool'),
            K.cast(K.ones_like(y_pred), 'bool')
        ], axis=1
    )

    fp_3d = K.concatenate(
        [
            K.cast(K.abs(y_true - K.ones_like(y_true)), 'bool'),
            K.cast(K.round(y_pred), 'bool'),
            K.cast(K.ones_like(y_pred), 'bool')
        ], axis=1
    )

    fn_3d = K.concatenate(
        [
            K.cast(y_true, 'bool'),
            K.cast(K.abs(K.round(y_pred) - K.ones_like(y_pred)), 'bool'),
            K.cast(K.ones_like(y_pred), 'bool')
        ], axis=1
    )

    tp = K.sum(K.cast(K.all(tp_3d, axis=1), 'int32'))
    fp = K.sum(K.cast(K.all(fp_3d, axis=1), 'int32'))
    fn = K.sum(K.cast(K.all(fn_3d, axis=1), 'int32'))

    precision = tp / (tp + fp)
    
    return precision


# get recall 
def recall(y_true, y_pred):
   
    tp_3d = K.concatenate(
        [
            K.cast(y_true, 'bool'),
            K.cast(K.round(y_pred), 'bool'),
            K.cast(K.ones_like(y_pred), 'bool')
        ], axis=1
    )

    fp_3d = K.concatenate(
        [
            K.cast(K.abs(y_true - K.ones_like(y_true)), 'bool'),
            K.cast(K.round(y_pred), 'bool'),
            K.cast(K.ones_like(y_pred), 'bool')
        ], axis=1
    )

    fn_3d = K.concatenate(
        [
            K.cast(y_true, 'bool'),
            K.cast(K.abs(K.round(y_pred) - K.ones_like(y_pred)), 'bool'),
            K.cast(K.ones_like(y_pred), 'bool')
        ], axis=1
    )

    tp = K.sum(K.cast(K.all(tp_3d, axis=1), 'int32'))
    fp = K.sum(K.cast(K.all(fp_3d, axis=1), 'int32'))
    fn = K.sum(K.cast(K.all(fn_3d, axis=1), 'int32'))

    recall = tp / (tp + fn)
    
    return recall


def tp(y_true, y_pred):
   
    tp_3d = K.concatenate(
        [
            K.cast(y_true, 'bool'),
            K.cast(K.round(y_pred), 'bool'),
            K.cast(K.ones_like(y_pred), 'bool')
        ], axis=1
    )

    
    tp = K.sum(K.cast(K.all(tp_3d, axis=1), 'int32'))
    
    
    return tp

def fp(y_true, y_pred):
   
    tp_3d = K.concatenate(
        [
            K.cast(y_true, 'bool'),
            K.cast(K.round(y_pred), 'bool'),
            K.cast(K.ones_like(y_pred), 'bool')
        ], axis=1
    )

    fp_3d = K.concatenate(
        [
            K.cast(K.abs(y_true - K.ones_like(y_true)), 'bool'),
            K.cast(K.round(y_pred), 'bool'),
            K.cast(K.ones_like(y_pred), 'bool')
        ], axis=1
    )

  

    fp = K.sum(K.cast(K.all(fp_3d, axis=1), 'int32'))

    
    return fp


def fn(y_true, y_pred):
   
    tp_3d = K.concatenate(
        [
            K.cast(y_true, 'bool'),
            K.cast(K.round(y_pred), 'bool'),
            K.cast(K.ones_like(y_pred), 'bool')
        ], axis=1
    )

    fp_3d = K.concatenate(
        [
            K.cast(K.abs(y_true - K.ones_like(y_true)), 'bool'),
            K.cast(K.round(y_pred), 'bool'),
            K.cast(K.ones_like(y_pred), 'bool')
        ], axis=1
    )

    fn_3d = K.concatenate(
        [
            K.cast(y_true, 'bool'),
            K.cast(K.abs(K.round(y_pred) - K.ones_like(y_pred)), 'bool'),
            K.cast(K.ones_like(y_pred), 'bool')
        ], axis=1
    )

   
    fn = K.sum(K.cast(K.all(fn_3d, axis=1), 'int32'))

    
    return fn






        # compile the CNN
classifier.compile(optimizer = 'Adadelta', loss = 'binary_crossentropy', metrics = ['accuracy',f1_score,Precision,recall,tp,fp,fn])



from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        zca_whitening=True,  
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        rotation_range=40)

test_datagen = ImageDataGenerator(rescale=1./255)



training_set = train_datagen.flow_from_directory(
        'data/Training',
        target_size=(128, 128),
        batch_size=20,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
          'data/Test',
        target_size=(128, 128),
        batch_size=20,
        class_mode='binary')




callback=keras.callbacks.ModelCheckpoint('my_model.h5', monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto')


history=classifier.fit_generator(
            training_set,
            steps_per_epoch=(832)/20,
            epochs=75,
            validation_data=test_set,
            validation_steps=(334)/20,
            callbacks = [callback])

#plot_model(classifier, to_file='model.png')

print(history.history.keys())
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['f1_score'])
plt.plot(history.history['val_f1_score'])
plt.title('model f measure')
plt.ylabel('f measure')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


plt.plot(history.history['Precision'])
plt.plot(history.history['val_Precision'])
plt.title('model precision')
plt.ylabel('Precision')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['recall'])
plt.plot(history.history['val_recall'])
plt.title('model recall')
plt.ylabel('recall')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

from keras.models import load_model
classifier.save('my_model.h5')
