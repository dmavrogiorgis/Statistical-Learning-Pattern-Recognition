from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
# Load data
fid = open('exam_scores_data1.txt', 'r')
lines = fid.readlines()
fid.close()

input_list = []
target_list = []

for line in lines:
    fields = line.rstrip().split(',')
    input_list.append([float(fields[0]), float(fields[1])])
    target_list.append([float(fields[2])]) 

X_all = np.array(input_list, dtype=np.float32) # Matrix of input features. shape = (num_examples, num_features)
Y_all = np.array(target_list, dtype=np.float32)   # Vector of target values. shape = (num_examples, )

N = X_all.shape[0] # Number of examples

# Normalize input data
# your code here
m  = np.mean(X_all)
s = np.std(X_all)

X_all = (X_all - m) / s

# Define the model
num_features = 2
output_depth = 1
batch_size = 8
learning_rate = 0.001


def sigmoid(z):
    # your code here
    return 1/(1 + np.exp(-z))

class NeuralNetwork:
    def __init__(self, input_depth, output_depth, learning_rate):
        self.W = np.sqrt(2.0/(input_depth + output_depth))*np.random.rand(input_depth, output_depth).astype(np.float32)
        self.b = np.zeros((output_depth, ), dtype=np.float32)   
        self.learning_rate = learning_rate

    def forward(self, x):
        # your code here
        return sigmoid(np.dot(x,self.W) + self.b)

    def backward(self, X, Y, Y_predicted):
        # your code here  
        B = X.shape[0]
        dJ_dZ = -Y + Y_predicted 
        dJ_dW = (1/B) * np.dot(dJ_dZ.T, X)
        dJ_db = (1/B) * np.sum(dJ_dZ)
        return dJ_dW,dJ_db

    def cross_entropy(self, Y, Y_predicted):
        # your code here
        return np.mean(-np.multiply(Y,np.log(Y_predicted))-np.multiply(1-Y,np.log(1-Y_predicted)))

    def update_weights(self, d_CE_d_W, d_CE_d_b):
        # your code here   
        self.W = self.W - np.dot(self.learning_rate,d_CE_d_W).reshape(2,1)
        self.b = self.b - self.learning_rate * d_CE_d_b

nn = NeuralNetwork(num_features, output_depth, learning_rate)

# Training the model
num_epochs = 400
num_batches = N - batch_size + 1

for epoch in range(num_epochs):
    epoch_loss = 0 
    for i in range(num_batches): # Sliding window of length = batch_size and shift = 1
        X = X_all[i:i+batch_size, :] 
        Y = Y_all[i:i+batch_size, :]

        Y_predicted = nn.forward(X) 
        batch_loss = nn.cross_entropy(Y, Y_predicted)   
        epoch_loss += batch_loss

        d_CE_d_W, d_CE_d_b = nn.backward(X, Y, Y_predicted) 
        nn.update_weights(d_CE_d_W, d_CE_d_b)

    epoch_loss /= num_batches
    print('epoch_loss = ', epoch_loss)


# Using the trained model to predict the probabilities of some examples and the compute the accuracy 

# Predict the normalized example [45, 85]
example = (np.array([[45, 85]], dtype=np.float32) - m)/s

print('Predicting the probabilities of example [45, 85]')
# your code here
example_prob = nn.forward(example)
print("Probability = %f \n" % example_prob)

Y_pred = nn.forward(X_all)
for i in range (np.size(Y_pred)):
        	if Y_pred[i] > 0.5:
        		Y_pred[i] = 1
        	else:
        		Y_pred[i] = 0

predicted_right = 0
for i in range(Y_all.shape[0]):
	if Y_all[i] == Y_pred[i]:
		predicted_right += 1

# Predict the accuracy of the training examples
accuracy_np = predicted_right / np.size(Y_all) # your code here

print('accuracy = ', accuracy_np) 

plt.figure()

plt.subplot(1,2,1)
plt.title("Original Dataset")
idx = np.argwhere(Y_all==1)
X1 = X_all[idx,0]
Y1 = X_all[idx,1]
idx = np.argwhere(Y_all==0)
X2 = X_all[idx,0]
Y2 = X_all[idx,1]
plt.scatter(X1,Y1,color='red')
plt.scatter(X2,Y2,color='blue')
plt.xlabel("Exam Grade 1")
plt.ylabel("Exam Grade 2")

plt.subplot(1,2,2)
plt.title("Neural Network Predictions")
idx = np.argwhere(Y_pred==1)
X1 = X_all[idx,0]
Y1 = X_all[idx,1]
idx = np.argwhere(Y_pred==0)
X2 = X_all[idx,0]
Y2 = X_all[idx,1]
plt.scatter(X1,Y1,color='red')
plt.scatter(X2,Y2,color='blue')
plt.xlabel("Exam Grade 1")
plt.ylabel("Exam Grade 2")
plt.show()
