from google.colab import drive
drive.mount('/content/drive')
!mkdir images

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import resize

target = []
images = []
flat_image = []
DATADIR = '/content/drive/MyDrive/MiniProject/Images'
Categ = ['Airplane','Bike','Car','Person']
for i in Categ:
  class_num = Categ.index(i)
  path = os.path.join(DATADIR,i)
  for img in os.listdir(path):
    img_array = imread(os.path.join(path,img))
    img_resized = resize(img_array,(200,200,3))
    flat_image.append(img_resized.flatten())
    images.append(img_resized)
    target.append(class_num)

flat_image = np.array(flat_image)
target = np.array(target)
images = np.array(images)

np.unique(target, return_counts = True)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(flat_image,target,random_state=32)

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

model = SVC()
param = {
    'kernel':['rbf'],
    'C':[10]
}

grid = GridSearchCV(model,param,cv=5)
grid.fit(x_train,y_train)

y_pred = grid.predict(x_test)

y_pred

y_test

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
accuracy_score(y_pred,y_test)

confusion_matrix(y_pred,y_test)

flattened = []
url = input('Enter a URL')
img = imread(url)
resized_img = resize(img,(200,200,3))
flattened.append(resized_img.flatten())
flattened = np.array(flattened)
print(img.shape)
plt.imshow(resized_img)
y_out = grid.predict(flattened)
y_out = Categ[y_out[0]]
print(f'Prediction: {y_out}')