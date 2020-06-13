import matplotlib.pyplot as plt 
from sklearn import datasets, svm, metrics 
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import train_test_split 

digits= datasets.load_digits()
_,axes=plt.subplots(2,5)
images_and_labels=list(zip(digits.images,digits.target))

for ax, (image,label) in zip(axes[0,:],images_and_labels[:5]):
    ax.set_axis_off()
    ax.imshow(image,cmap=plt.cm.gray_r,interpolation='nearest')
    ax.set_title("Training: %i" %label)

n_samples= len(digits.images)
data=digits.images.reshape((n_samples,-1))

X_train,X_test,y_train,y_test=train_test_split(data,digits.target,test_size=0.5,shuffle=False)
classifier=svm.SVC(kernel='rbf',gamma=0.001)
classifier.fit(X_train,y_train)

predicted = classifier.predict(X_test)

test_data= X_test.reshape((len(X_test),8,8))
images_and_labels = list(zip(test_data,predicted))

for ax,(image,prediction) in zip(axes[1,:],images_and_labels[:5]):
    ax.set_axis_off()
    ax.imshow(image,cmap=plt.cm.gray_r,interpolation='nearest')
    ax.set_title('Predict: %i' %prediction)

print("SVM 분류 결과 %s:\n%s\n" %(classifier,metrics.classification_report(y_test,predicted)))

disp = metrics.plot_confusion_matrix(classifier,X_test,y_test)
disp.figure_.suptitle("Confusion Matrix")
print("혼동 행렬:\n%s" %disp.confusion_matrix)
print("정확도:",accuracy_score(y_test,predicted))

plt.show()