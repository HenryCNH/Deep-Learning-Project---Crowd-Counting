import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame
import operator
import random
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from keras import Input, Model, Sequential
from tensorflow.keras.layers import Conv2D,Dense,concatenate,Activation,Dropout,Input
from tensorflow.keras.models import Model
import time

AUTOTUNE = tf.data.AUTOTUNE

image_link = r"address_of_image_in_numpy_format"
labels_link = r"labels_in_csv"

def import_dataset(x,y):
    image_dataset = np.load(x)
    labels_dataset = pd.read_csv(y)
    final_labels_dataset = labels_dataset['count']
    print(image_dataset[0])
    print(image_dataset.shape)
    print(labels_dataset.shape)
    print(final_labels_dataset.head(5))
    return(image_dataset,labels_dataset,final_labels_dataset)

image_dataset, labels_dataset,final_labels_dataset = import_dataset(image_link, labels_link)

#Creating a tensorflow data set
tf_dataset = tf.data.Dataset.from_tensor_slices((image_dataset,final_labels_dataset))
print(len(tf_dataset))

#Showing images of dataset
fig=plt.figure(figsize=(10,10))
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(image_dataset[i])
    plt.title("Number of People : {0}".format(labels_dataset['count'][i]))
plt.show()

#Exploratory Data Analysis
def EDA(y):
    print(y['count'].describe())
    y['count'].hist(bins=40)
    plt.xlabel('Number of People')
    plt.ylabel('Number of Images')
    plt.title('Distribution of number of people')
    plt.show()
    return
EDA(labels_dataset)


#Splitting training data and validation data into 0.8 : 0.2 which is 1600 : 400
def splitting_dataset(d):
    print('Number of total datasets :')
    print(tf.data.experimental.cardinality(d).numpy())
    X_train_ds=d.take(1600)
    X_val_ds=d.skip(1600)
    X_train_org= d.take(1600)
    X_test_org= d.skip(1600)
    print('Number of training datasets :')
    print(tf.data.experimental.cardinality(X_train_ds).numpy())
    print('Number of validation datasets :')
    print(tf.data.experimental.cardinality(X_val_ds).numpy())
    return(X_train_ds, X_val_ds, X_train_org, X_test_org)
X_train_ds, X_val_ds, X_train_org, X_test_org = splitting_dataset(tf_dataset)



#Convert number of people from tensor to uint8 for later result comparison use
number_of_people_tensor_train = []

number_of_people_tensor_test = []
for i in X_train_ds:
    number_of_people_tensor_train.append(i[1])
for i in X_val_ds:
    number_of_people_tensor_test.append(i[1])

number_of_people_unit8_train = []
for i in range(0, 1600):
    number_of_people_unit8_train.append(number_of_people_tensor_train[i].numpy().astype("uint8"))

number_of_people_unit8_test = []
for i in range(0, 400):
    number_of_people_unit8_test.append(number_of_people_tensor_test[i].numpy().astype("uint8"))



#Conduct Data Preprocessing
seed = (1, 2)
def augmentation_process(image, count):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, [400, 400])
    image = tf.image.stateless_random_flip_left_right(image, seed)
    image = tf.image.stateless_random_flip_up_down(image, seed)
    image = tf.image.stateless_random_brightness(image, max_delta=32.0 / 255.0, seed=seed)
    image = tf.image.stateless_random_saturation(image, lower=0.5, upper=1.5, seed=seed)
    return image, count

X_train_ds = X_train_ds.map(augmentation_process, num_parallel_calls=AUTOTUNE)
X_val_ds = X_val_ds.map(augmentation_process, num_parallel_calls=AUTOTUNE)

def improve_performance (dataset):
    dataset = dataset.cache()
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(2)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    return dataset

input_shape=(400,400,3)

X_train_ds = improve_performance(X_train_ds)
X_val_ds = improve_performance(X_val_ds)


#Creating CNN Model and training the model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(1))

model.compile(loss=tf.keras.losses.Huber(), optimizer=keras.optimizers.Nadam(3e-4), metrics=['mae','mse'])
model.summary()
callbacks = [keras.callbacks.ModelCheckpoint("Weights_v2_of_epoch_{epoch}.keras")]
csv_logger = tf.keras.callbacks.CSVLogger("logger_v2_csv.csv", separator=",", append=False)
Model= model.fit(X_train_ds,
          epochs=30,
          verbose=1,
          validation_data=(X_val_ds),
        callbacks=[csv_logger,callbacks])

y_train_result=model.predict(X_train_ds)
y_test_result=model.predict(X_val_ds)


#Showing graph of mae vs val_mae and mse vs val_mse of each epoch
epochs=30
def plt_dynamic(xaxis, y_validation, y_train, ax, colors=['b']):
    ax.plot(xaxis, y_validation, 'b', label="Validation")
    ax.plot(xaxis, y_train, 'r', label="Train")
    plt.legend()
    plt.grid()
    plt.show()

fig,ax = plt.subplots(1,1)
ax.set_xlabel('epoch')
ax.set_ylabel('Mean Squared error/Loss')
xaxis = list(range(1,epochs+1))
y_validation = Model.history['val_mse']
y_train = Model.history['mse']
plt_dynamic(xaxis, y_validation, y_train, ax)

fig,ax = plt.subplots(1,1)
ax.set_xlabel('epoch')
ax.set_ylabel('Mean Absolute error/Loss')
xaxis = list(range(1,epochs+1))
y_validation = Model.history['val_mae']
y_train = Model.history['mae']
plt_dynamic(xaxis, y_validation, y_train, ax)



#add to the dataframe
Model_Train_results = pd.DataFrame()
Model_Train_results['Train_Actual']= number_of_people_unit8_train
Model_Train_results['Train_Predicted']= np.round(y_train_result)
Model_Train_results['Train_Difference(actual-predicted)']  = Model_Train_results['Train_Actual'] - Model_Train_results['Train_Predicted']
Model_Train_results['Train_Difference(actual-predicted)'] = Model_Train_results['Train_Difference(actual-predicted)'].abs()
print(Model_Train_results.head(10))

Model_Test_results = pd.DataFrame()
Model_Test_results['Test_Actual']= number_of_people_unit8_test
Model_Test_results['Test_Predicted']= np.round(y_test_result)
Model_Test_results['Test_Difference(actual-predicted)']  = Model_Test_results['Test_Actual'] - Model_Test_results['Test_Predicted']
Model_Test_results['Test_Difference(actual-predicted)'] =Model_Test_results['Test_Difference(actual-predicted)'].abs()
print(Model_Test_results.head(10))



# Print Actual Train vs Predict
plt.figure(figsize=(20, 20))
i = 0
for element in X_train_org:
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(element[0].numpy().astype("uint8"))
    plt.title(
        "Actual People Count: " + str(Model_Train_results['Train_Actual'][i]) + "\n Predicted People Count: " + str(
            Model_Train_results['Train_Predicted'][i]))
    plt.axis("off")
    i += 1
    if i > 8:
        break
plt.show()


# Print Actual Test vs Predict
plt.figure(figsize=(20, 20))
i = 0
for element in X_test_org:
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(element[0].numpy().astype("uint8"))
    plt.title("Actual People Count: " + str(Model_Test_results['Test_Actual'][i]) + "\n Predicted People Count: " + str(
        Model_Test_results['Test_Predicted'][i]))
    plt.axis("off")
    i += 1
    if i > 8:
        break
plt.show()





















