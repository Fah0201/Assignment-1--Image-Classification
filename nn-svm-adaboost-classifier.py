import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from skimage.feature import hog
from tqdm import trange
from tqdm import tqdm 
import matplotlib.pyplot as plt

# Function to read images and labels from a directory
def read_images(directory):
    images = []
    labels = []
    with open(directory, 'r') as f:
        for line in f:
            image = cv2.imread(line.split(' ')[0])
            image = cv2.resize(image, (128, 128)) # Resize image to 256x256
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            images.append(image)
            labels.append(line.split(' ')[1])
    return images, labels

# Function to extract HOG features from an image
def extract_features(image):
    features = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3), visualize=False, transform_sqrt=True)
    return features

# Function to calculate top1 & top5
def top1_5_score(truth_y,test_x,model):
    prd_y = model.predict_proba(test_x)
    label = model.classes_
    top1_score = 0
    top5_score = 0
    for i in tqdm(range(len(truth_y))):
        top5_ans = np.argpartition(prd_y[i], -5)[-5:]
        print(top5_ans)
        if str(int(truth_y[i])) in label[top5_ans]:
            top5_score = top5_score + 1
        if str(int(truth_y[i])) == label[np.argmax(prd_y[i])]:
            top1_score = top1_score + 1
    print(top1_score/len(truth_y) , top5_score/len(truth_y))
    return top1_score/len(truth_y) , top5_score/len(truth_y)

# Function to train and test a classifier
def train_test_classifier(trainX, trainY, testX, testY, valX, valY):
    
    # Extract HOG features from the images
    print('Extract HOG features')
    
    X_train_features=[]
    for i in trange(len(trainX)):
        X_train_features.append(extract_features(trainX[i]))
    X_test_features=[]
    for i in trange(len(testX)):
        X_test_features.append(extract_features(testX[i]))
    X_val_features=[]
    for i in trange(len(valX)):
        X_val_features.append(extract_features(valX[i]))
    X_train_features = np.asarray(X_train_features)
    X_test_features = np.asarray(X_test_features)
    X_val_features = np.asarray(X_val_features)
    
    # Train an nn model classifier
    print('Train an nn model classifier')
    nn_clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=200, alpha=1e-4, solver='sgd', verbose=10, tol=1e-4, random_state=1,learning_rate_init=.1)
    nn = nn_clf.fit(X_train_features, trainY)
    nn_score = nn_clf.score(X_test_features, testY)
    print('NN model classifier score:', nn_score)
    nn_val_score = nn_clf.score(X_val_features, valY)
    print('NN model classifier val score:', nn_val_score)
    nn_test_score=[]
    for i in trange(1, X_test_features.shape[0], 45):
        nn_test_score.append(nn_clf.score(X_test_features[:i], testY[:i]))
    nn_val_score=[]
    for i in trange(1, X_val_features.shape[0], 45):
        nn_val_score.append(nn_clf.score(X_val_features[:i], valY[:i]))
    plt.plot(nn_test_score, label='nn Test Score')
    plt.plot(nn_val_score, label='nn Score')
    plt.legend()
    plt.show()

    # Train a SVM classifier
    print('Train a SVM classifier')
    svm_clf = SVC(kernel='linear',probability=True)
    svm = svm_clf.fit(X_train_features, trainY)
    svm_score = svm_clf.score(X_test_features, testY)
    print('SVM classifier score:', svm_score)
    svm_val_score = svm_clf.score(X_val_features, valY)
    print('SVM classifier val score:', svm_val_score)
    svm_test_score=[]
    for i in trange(1, X_test_features.shape[0], 45):
        svm_test_score.append(svm_clf.score(X_test_features[:i], testY[:i]))
    svm_val_score=[]
    for i in trange(1, X_val_features.shape[0], 45):
        svm_val_score.append(svm_clf.score(X_val_features[:i], valY[:i]))
    plt.plot(svm_test_score, label='SVM Test Score')
    plt.plot(svm_val_score, label='SVM Val Score')
    plt.legend()
    plt.show()    
       
    # Train an AdaBoost classifier
    print('Train an AdaBoost classifier')
    ada_clf = AdaBoostClassifier(n_estimators=100)
    ada = ada_clf.fit(X_train_features, trainY)
    ada_score = ada_clf.score(X_test_features, testY)
    print('AdaBoost classifier score:', ada_score)
    ada_val_score = ada_clf.score(X_val_features, valY)
    print('AdaBoost classifier val score:', ada_val_score)
    ada_test_score=[]
    for i in trange(1, X_test_features.shape[0], 45):
        ada_test_score.append(ada_clf.score(X_test_features[:i], testY[:i]))
    ada_val_score=[]
    for i in trange(1, X_val_features.shape[0], 45):
        ada_val_score.append(ada_clf.score(X_val_features[:i], valY[:i]))
    plt.plot(ada_test_score, label='AdaBoost Test Score')
    plt.plot(ada_val_score, label='AdaBoost Score')
    plt.legend()
    plt.show()
    
    nn_test_result = top1_5_score(testY, X_test_features, nn)
    nn_val_result = top1_5_score(valY, X_val_features, nn)
    ada_test_result = top1_5_score(testY, X_test_features, ada)
    ada_val_result = top1_5_score(valY, X_val_features, ada)
    svm_test_result = top1_5_score(testY, X_test_features, svm)
    svm_val_result = top1_5_score(valY, X_val_features, svm)
    
    return svm, ada, nn, svm_test_result, svm_val_result, ada_test_result, ada_val_result, nn_test_result, nn_val_result

if __name__ == '__main__':
    # Read the images and labels from the directory
    trainImages, trainLabels = read_images('test.txt')
    testImages, testLabels = read_images('test.txt')
    valImages, valLabels = read_images('val.txt')
    print('Read all files.')

    # Train and test the classifiers
    svm, ada, nn, svm_test_result, svm_val_result, ada_test_result, ada_val_result, nn_test_result, nn_val_result = train_test_classifier(trainImages, trainLabels, testImages, testLabels, valImages, valLabels)
