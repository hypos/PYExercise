import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.preprocessing import LabelBinarizer
from sklearn.cross_validation import train_test_split
from NNetwork import NerualNetwork

digits=load_digits()

# import pylab as pl
# pl.gray()
# pl.matshow(digits.images[0])
# pl.matshow(digits.images[1])
# pl.matshow(digits.images[2])
# pl.matshow(digits.images[3])
# pl.matshow(digits.images[4])
# pl.matshow(digits.images[5])
# pl.matshow(digits.images[6])
# pl.matshow(digits.images[7])
# pl.matshow(digits.images[8])
# pl.matshow(digits.images[9])
# pl.show()

x=digits.data
print(x)
y=digits.target
print(y)
x-=x.min()
x/=x.max()

nn=NerualNetwork([64,100,10],'logistic')
x_train,x_test,y_train,y_test=train_test_split(x,y)
labels_train=LabelBinarizer().fit_transform(y_train)
labels_test=LabelBinarizer().fit_transform(y_test)
nn.fit(x_train,labels_train,epochs=3000)
predictions=[]
for i in range(x_test.shape[0]):
    o=nn.predict(x_test[i])
    predictions.append(np.argmax(o))

print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
