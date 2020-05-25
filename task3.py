
import keras
from keras.models import Sequential
from keras.layers import Dense,Flatten,MaxPooling2D,Convolution2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
import smtplib, ssl


model=Sequential()
model.add(Convolution2D(filters=32,kernel_size=(3,3),activation='relu',input_shape=(64,64,3)))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(units=128,activation='relu'))
model.add(Dense(units=1,activation='sigmoid'))
model.summary()
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


f=open("requirement.txt","r+")
epoch=int((f.readline())
f.close()


train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        '/mlops/task3/animals/valid/',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')
validation_generator = test_datagen.flow_from_directory(
        '/mlops/task3/animals/valid/',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')
history=model.fit(
        train_generator,
        steps_per_epoch=1,
        epochs=epoch,
        validation_data=validation_generator,
        validation_steps=800)




l=[]
l=history.history['accuracy']



n=len(l)
accuracy=l[n-1]
if accuracy<0.8:
    f1=open("report.txt","w")
    f1.write(str(accuracy))
    f1.close()

    f2=open("requirement.txt","w")
    epoch=int(epoch)+1
    f2.write(str(epoch))
    f2.close
else:
    model.save("model.h5")




