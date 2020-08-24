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
import matplotlib.pyplot as plt







classifier = Sequential()
    
        # Step1 Convolution
    
classifier.add(Convolution2D(32,3,3,input_shape=(128,128,3),activation = 'relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Convolution2D(32,3,3,input_shape=(128,128,3),activation = 'relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Convolution2D(32,3,3,input_shape=(128,128,3),activation = 'relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

        
        # Step 3 Flattening 
        
classifier.add(Flatten())
    
        #step 4 Full Connection 
        
classifier.add(Dense(64, activation = 'relu'))
classifier.add(Dense(64, activation = 'relu'))


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

def f1_score(y_true, y_pred, beta=1):
  
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = Precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score

#Get Precision
def Precision(y_true, y_pred):
    """Precision metric.
    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    """Recall metric.
    Only computes a batch-wise average of recall.
    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall



def tp(y_true, y_pred):
   
    tp_3d = K.concatenate(
        [
            K.cast(y_true, 'int32'),
            K.cast(K.round(y_pred), 'int32'),
            K.cast(K.ones_like(y_pred), 'int32')
        ], axis=1
    )

    
    tp = K.sum(K.cast(K.all(tp_3d, axis=1), 'int32'))
    
    
    return tp

def fp(y_true, y_pred):
   
    tp_3d = K.concatenate(
        [
            K.cast(y_true, 'int32'),
            K.cast(K.round(y_pred), 'int32'),
            K.cast(K.ones_like(y_pred), 'int32')
        ], axis=1
    )

    fp_3d = K.concatenate(
        [
            K.cast(K.abs(y_true - K.ones_like(y_true)), 'int32'),
            K.cast(K.round(y_pred), 'int32'),
            K.cast(K.ones_like(y_pred), 'int32')
        ], axis=1
    )

  

    fp = K.sum(K.cast(K.all(fp_3d, axis=1), 'int32'))

    
    return fp


def fn(y_true, y_pred):
   
    tp_3d = K.concatenate(
        [
            K.cast(y_true, 'int32'),
            K.cast(K.round(y_pred), 'int32'),
            K.cast(K.ones_like(y_pred), 'int32')
        ], axis=1
    )

    fp_3d = K.concatenate(
        [
            K.cast(K.abs(y_true - K.ones_like(y_true)), 'int32'),
            K.cast(K.round(y_pred), 'int32'),
            K.cast(K.ones_like(y_pred), 'int32')
        ], axis=1
    )

    fn_3d = K.concatenate(
        [
            K.cast(y_true, 'int32'),
            K.cast(K.abs(K.round(y_pred) - K.ones_like(y_pred)), 'int32'),
            K.cast(K.ones_like(y_pred), 'int32')
        ], axis=1
    )

   
    fn = K.sum(K.cast(K.all(fn_3d, axis=1), 'int32'))

    
    return fn


        # compile the CNN
classifier.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics = ['accuracy',f1_score,Precision,recall,tp,fp,fn])



from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator( zca_whitening=True,  
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        rotation_range=40
       )

test_datagen = ImageDataGenerator(rescale=1./255)



training_set = train_datagen.flow_from_directory(
        'data/Training',
        target_size=(128, 128),
        batch_size=70,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
          'data/Test',
        target_size=(128, 128),
        batch_size=70,
        class_mode='binary')




callback=keras.callbacks.ModelCheckpoint('my_model.h5', monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto')


history=classifier.fit_generator(
            training_set,
            steps_per_epoch=(900)/70,
            epochs=100,
            validation_data=test_set,
            validation_steps=(375)/70,
            callbacks = [callback])


import csv

with open('TrialResult.csv','w') as f:
    writer = csv.writer(f)
    for key,values in history.history.items():
        o = []
        o.append(key)
        for v in values:
            o.append(v)
        writer.writerow(o)


from keras.models import load_model
classifier.save('my_model.h5')




print(history.history.keys())
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Blind Person Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch Number')
plt.legend(['Training Set', 'Test Set'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Blind Person Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch Number')
plt.legend(['Training Set', 'Test Set'], loc='upper left')
plt.show()

plt.plot(history.history['f1_score'])
plt.plot(history.history['val_f1_score'])
plt.title('Blind Person Model F1 Score')
plt.ylabel('F1 Score')
plt.xlabel('Epoch Number')
plt.legend(['Training Set', 'Test Set'], loc='upper left')
plt.show()


plt.plot(history.history['recall'])
plt.plot(history.history['val_recall'])
plt.title('Blind Person Model Recall')
plt.ylabel('Recall')
plt.xlabel('Epoch Number')
plt.legend(['Training Set', 'Test Set'], loc='upper left')
plt.show()

plt.plot(history.history['Precision'])
plt.plot(history.history['val_Precision'])
plt.title('Blind Person Model Precision')
plt.ylabel('Precision')
plt.xlabel('Epoch Number')
plt.legend(['Training Set', 'Test Set'], loc='upper left')
plt.show()

plt.plot(history.history['tp'])
plt.plot(history.history['val_tp'])
plt.title('Blind Person Model True Positives')
plt.ylabel('True Positives')
plt.xlabel('Epoch Number')
plt.legend(['Training Set', 'Test Set'], loc='upper left')
plt.show()

plt.plot(history.history['fn'])
plt.plot(history.history['val_fn'])
plt.title('Blind Person Model False Negatives')
plt.ylabel('False Negatives')
plt.xlabel('Epoch Number')
plt.legend(['Training Set', 'Test Set'], loc='upper left')
plt.show()

plt.plot(history.history['fp'])
plt.plot(history.history['val_fp'])
plt.title('Blind Person Model False Positives')
plt.ylabel('False Positives')
plt.xlabel('Epoch Number')
plt.legend(['Training Set', 'Test Set'], loc='upper left')
plt.show()





classifier.summary()

for layer in model.layers:
   weights = layer.get_weights()

from keras.utils import plot_model
plot_model(classifier, to_file='model_plot.png', show_shapes=True, show_layer_names=True)


