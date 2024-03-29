clear all
close all

load('exam_scores_data1.txt')

X_all = exam_scores_data1(:, 1:2); % input
Y_all = exam_scores_data1(:, 3); % target

N = size(X_all, 1); % Number of examples

% Normalize input data (X_all)
% your code here


% Define the model
num_features = 2;
input_depth = num_features;
output_depth = 1;
batch_size = 8;
learning_rate = 0.001;

% Define the weights
W = normrnd(0, sqrt(2.0/(input_depth + output_depth)), input_depth, output_depth); 
b = zeros(output_depth, 1);   

% Training the model
num_epochs = 55;
num_batches = N - batch_size;

for epoch = 1:num_epochs
    epoch_loss = 0; 
    for i = 1:num_batches % Sliding window of length = batch_size and shift = 1
        X = X_all(i:i+batch_size, :); 
        Y = Y_all(i:i+batch_size, :);

        Y_predicted = forward(X, W, b);
        batch_loss = cross_entropy(Y, Y_predicted);   
        epoch_loss = epoch_loss + batch_loss;

        [d_CE_d_W, d_CE_d_b] = backward(X, Y, Y_predicted); 

        % Update weights
        % your code here
    end 

    epoch_loss = epoch_loss/num_batches;
    disp(strcat('epoch_loss = ', num2str(epoch_loss)))
end

% Using the trained model to predict the probabilities of some examples and the compute the accuracy 

% Predict the normalized example [45, 85]
example = ([45, 85] - m)./s

disp('Predicting the probabilities of example [45, 85]')
% your code here

% Predict the accuracy of the training examples
accuracy = ... % your code here
 


