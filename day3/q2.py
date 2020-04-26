#!/usr/bin/env python
import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

(train_images,train_labels),(test_images,test_labels)=mnist.load_data()
print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(test_labels.shape)

# to find number of classification

categories=np.unique(train_labels)#categories in y
ncat=len(categories)

'''
#Display te first image in training data

plt.subplot(122)
plt.imshow(train_images[20,:],cmap='gray')
plt.title("Ground Trurh:{}".format(test_labels[20]))
plt.show()



#Display the first image in testing data

plt.subplot(122)
plt.imshow(test_images[20,:],cmap='gray')
plt.title("Ground Truth:{}".format(test_labels[20]))
plt.show()
'''

dimdata=np.prod(train_images.shape[1:])
X_train=train_images.reshape(train_images.shape[0],dimdata)
X_test=test_images.reshape(test_images.shape[0],dimdata)

#changing the data to float
X_train=X_train.astype('float32')
X_test=X_test.astype('float32')

#Scale the data lie between 0 to 1

X_train=X_train/255
X_test=X_test/255

#Change the labels from integer to categorical data
y_train=to_categorical(train_labels)
y_test=to_categorical(test_labels)


model=Sequential()
model.add(Dense(512,activation='relu',input_shape=(dimdata,)))
model.add(Dense(512,activation='relu'))
model.add(Dense(ncat,activation='relu'))

model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
history=model.fit(X_train,y_train,batch_size=256,epochs=5,verbose=1,validation_data=(X_test,y_test))

[test_loss,test_acc]=model.evaluate(X_test,y_test)
print("Evaluationresult on Test Data:Loss={},accuracy={}".format(test_loss,test_acc))
