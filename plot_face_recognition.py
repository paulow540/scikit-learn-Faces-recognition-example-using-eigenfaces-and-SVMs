"""                         FACE RECOGNITION WITH SCIKIT-LEARN IMAGES

SOURCE: We will use the Labeled Faces in the Wild dataset, which consists of several thousand collated photos of various public figures


DATA FOR THE PROJECT 
The Data is from Scikit-Learn datasets for fetch_lfw_people
Now let check how we can use supervised learning algorithm to solve Facial recognition problems for scikit-learn datasets for the fetch_lfw_people

SUPPORT VECTOR MACHINE CLASSIFICATION

What is the Support Vector Machine?
“Support Vector Machine” (SVM) is a supervised machine learning algorithm that can be used for both classification and regression challenges. However, it is mostly used in classification problems. In the SVM algorithm, we plot each data item as a point in n-dimensional space (where n is a number of features you have) with the value of each feature being the value of a particular coordinate. Then, we perform classification by finding the hyper-plane that differentiates the two classes very well, so we will use Support Vector Machine to solve the problem.
For use to work with Support Vector Machine, we will import all the modules and the images from sklearn.datasets.

import numpy as np
NumPy is a Python library used for working with arrays. It also has functions for working in domain of linear algebra, fourier transform, and matrices. 


import pandas as pd
Pandas is a Python library. Pandas is used to analyze data. Learning by Reading. We have created 14 tutorial pages for you to learn more about Pandas.

import matplotlib.pyplot as plt
Matplotlib.pyplot is a state-based interface to matplotlib. It provides an implicit, MATLAB-like, way of plotting. It also opens figures on your screen, and acts as the figure GUI manager. pyplot is mainly intended for interactive plots and simple cases of programmatic plot generation
%matplotlib inline


import seaborn as sns
Seaborn is a Python data visualization library based on matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics.

"""

#####################NOW LET IMPORT OTHERS MODULES AND THE DATASET########################3
## we imported time function fron a time module
from time import time

## we imported matplotlib.pyplot 
import matplotlib.pyplot

##we imported train_test_split from sklearn.model_selection=== train_test_split it is used to split the train and test dataset  
from sklearn.model_selection import train_test_split


## we imported RandomizedSearchCV # Selecting Best Models Using Randomized Search
# Problem: You want a computationally cheaper method than exhaustive search to select the best
# model.
# Solution: Use scikit-learn’s RandomizedSearchCV:
from sklearn.model_selection import RandomizedSearchCV  

### we imported fetch_lfw_people dataset from sklearn.datasets
from sklearn.datasets import fetch_lfw_people 

### Creating a Text Report of Evaluation Metrics
# Problem: You want a quick description of a classifier’s performance.
# Solution: Use scikit-learn’s classification_report:
from sklearn.metrics import classification_report 

## we imported ConfusionMatrixDisplay from sklearn.metrics
from sklearn.metrics import ConfusionMatrixDisplay


#### Standardizing a Feature
# Problem: You want to transform a feature to have a mean of 0 and a standard deviation of 1.
# Solution: scikit-learn’s StandardScaler performs both transformations:
from sklearn.preprocessing import StandardScaler

### Reducing Features Using Principal Components
# Problem:Given a set of features, you want to reduce the number of features while retaining the
# variance in the data.
# Solution: Use principal component analysis with scikit’s PCA:
from sklearn.decomposition import PCA 

#### Training a Linear Classifer
# Problem: You need to train a model to classify observations.
# Solution: Use a support vector classifier (SVC) to find the hyperplane that maximizes the mar‐
# gins between the classes:
from sklearn.svm import SVC  
from sklearn.utils.fixes import loguniform




lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

# introspect the images arrays to find the shapes (for plotting)
n_samples, h, w = lfw_people.images.shape

# for machine learning we use the 2 data directly (as relative pixel
# positions info is ignored by this model)


## we assign all the dataset to this variable X
X = lfw_people.data

## we assign all the dataset shape to this variable n_features
n_features = X.shape[1]

# the label to predict is the id of the person

## Now this is the code for the target, all the lfw_people.target dataset is assign to a variable y
y = lfw_people.target

## This code is assigning all the target names to variable target_names
target_names = lfw_people.target_names

##This line of code is getting the first index of the target names and is assigned to a variable called n_classes
n_classes = target_names.shape[0]


##Printing the Total dataset size and persentage of the data sample, dataset features and dataset classes

print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)



###The below line of code explain how to split into a training set and a test and keep 25% of the data for testing with a function train_test_split.

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)


## Now let use StandardScaler to  transform a feature to have a mean of 0 and a standard deviation of 1 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


## Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
# dataset): unsupervised feature extraction / dimensionality reduction

n_components = 150

## Now let Extracte the top %d eigenfaces from %d faces the value of the componenets and the first index of the shape of the train data
print(
    "Extracting the top %d eigenfaces from %d faces" % (n_components, X_train.shape[0])
)

## let check for the time in which the model will work on the dataset
t0 = time()

####To reduce the number of features while retaining the variance in the data
pca = PCA(n_components=n_components, svd_solver="randomized", whiten=True).fit(X_train)
print("done in %0.3fs" % (time() - t0))


##Let reshape the values we get from the pca components
eigenfaces = pca.components_.reshape((n_components, h, w))


## thisline of codes print and transform the traing and testing data
print("Projecting the input data on the eigenfaces orthonormal basis")
t0 = time()
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print("done in %0.3fs" % (time() - t0))


## LET WORK ON THE SVM MODEL TO TREAT THE TRAING DATA AND TEST THE DATA AS WELL
# Train a SVM classification model

print("Fitting the classifier to the training set")
t0 = time()
param_grid = {
    "C": loguniform(1e3, 1e5),
    "gamma": loguniform(1e-4, 1e-1),
}


###Features and target configuration Create a pipeline of scalars and standard models for five 
# different regressors. Fit all models to training data Obtain the cross-validation mean on the training set for all negative mean squared error models Choose the model with the best cross-validation score Ride the best model on the training set Now let’s implement all the steps
#  mentioned above to train a machine learning model for the task of Diamond Price Prediction:
clf = RandomizedSearchCV(
    SVC(kernel="rbf", class_weight="balanced"), param_grid, n_iter=10
)

##let use our Pipleline to fit and check the best estimator
clf = clf.fit(X_train_pca, y_train)
print("done in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)



# Quantitative evaluation of the model quality on the test set

print("Predicting people's names on the test set")
t0 = time()

##Let predict the best estimator we get from the model
y_pred = clf.predict(X_test_pca)
print("done in %0.3fs" % (time() - t0))


###let use the model now to predict the value we get from the predictive ans the testing datadset
print(classification_report(y_test, y_pred, target_names=target_names))
ConfusionMatrixDisplay.from_estimator(
    clf, X_test_pca, y_test, display_labels=target_names, xticks_rotation="vertical"
)
plt.tight_layout()
plt.show()



# Qualitative evaluation of the predictions using matplotlib
##This line of code show a function in which it use use to plot all the images data with imshow
##Imshow is a model inscikit learn in which we use to plot any data as an image
def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=0.01, right=0.99, top=0.90, hspace=0.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())



# plot the result of the prediction on a portion of the test set
##These lines of code help us plot the prediction and the testing values and the prediction names
def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(" ", 1)[-1]
    true_name = target_names[y_test[i]].rsplit(" ", 1)[-1]
    return "predicted: %s\ntrue:      %s" % (pred_name, true_name)


prediction_titles = [
    title(y_pred, y_test, target_names, i) for i in range(y_pred.shape[0])
]

##plotting the test and predicted values
plot_gallery(X_test, prediction_titles, h, w)

# plot the gallery of the most significative eigenfaces

eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, h, w)

##we use plt.show() to visualize our plot if we are not working on a jupyter notebook
plt.show()


