import keras
from keras.models import Sequential
from keras.layers import Dense,Flatten,Activation
accuracy=90
f=open("report.txt","w")
f.write(str(accuracy))