#!/usr/bin/env python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split



from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import to_categorical

cancer_data=load_breast_cancer()
X=cancer_data.data
y_=cancer_data.target.reshape(-1,1)#Convert data to a single column
y=to_categorical(y_)

model=Sequential()

model.add(Dense(40,input_shape=(30,),activation='relu',name='fc1'))
model.add(Dense(40,activation='relu',name='fc2'))
model.add(Dense(2,activation='relu',name='output'))

#Adam optimizer with laerning rate 0.001

optimizer=Adam(lr=0.001)
model.compile(optimizer,loss='mean_squared_error',metrics=['mse'])

print('Neural network model summary:')
print(model.summary)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

#train the model

model.fit(X_train,y_train,verbose=1,batch_size=200,epochs=150)

results=model.evaluate(X_test,y_test)

print('Final test set loss:{:4f}'.format(results[0]))
print('Final test set accuracy:{:4f}'.format(results[1]))


