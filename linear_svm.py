import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm

# linear data
X = np.array([1, 2.4, 3.1, 3.1, 5, 6, 3, 2.5, 8, 1, 8, 8, 8.7, 1.3, 6.5, 7.7, 6.1])
y = np.array([2, 1, 2.4, 4.5, 6.2, 7, 3, 2.8, 8, 0.4, 11, 10, 9.4, 4.5, 5, 9, 8.5])

# show unclassified data
# plt.scatter(X, y)
# plt.show()

# shaping data for training the model
training_X = np.vstack((X,y)).T
training_y = np.where(y <= 5, 0, 1)

# Initializing the model
model = svm.SVC(kernel = 'linear', C = 1.0)

# train the model
model.fit(training_X, training_y)

# get the weight values for the linear equation from the trained SVM model
w = model.coef_[0]

# get the y-offset for the linear equation
a = -w[0] / w[1]

# make the x-axis space for the data points
XX = np.linspace(0, 13)

# get the y-values to plot the decision boundary
yy = a * XX - model.intercept_[0] / w[1]

# plot the decision boundary
plt.plot(XX, yy, 'k-')

# show the plot visually
plt.scatter(training_X[training_y == 0][:, 0], training_X[training_y == 0][:, 1], c='blue', label='Class 0')
plt.scatter(training_X[training_y == 1][:, 0], training_X[training_y == 1][:, 1], c='red', label='Class 1')

plt.legend()
plt.show()

