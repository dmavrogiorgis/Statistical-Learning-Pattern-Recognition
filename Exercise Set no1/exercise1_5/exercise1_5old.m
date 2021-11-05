close all;
clear;
clc;

data_file = './data/mnist.mat';

data = load(data_file);

% Read the train data
[train_C1_indices, train_C2_indices,train_C1_images,train_C2_images] = read_data(data.trainX,data.trainY.');

% Read the test data
[test_C1_indices, test_C2_indices,test_C1_images,test_C2_images] = read_data(data.testX,data.testY.');


%% Compute Aspect Ratio

aRatio_C1 = zeros(1, size(train_C1_images,1));
aRatio_C2 = zeros(1, size(train_C2_images,1));

for i = 1:size(train_C1_images)
  aRatio_C1(i) = computeAspectRatio(train_C1_images(i,:,:));
end

for i = 1:size(train_C2_images)
  aRatio_C2(i) = computeAspectRatio(train_C2_images(i,:,:));
end

% Compute the aspect ratios of all images and store the value of the i-th image in aRatios(i)

minAspectRatio = min([min(aRatio_C1) min(aRatio_C2)]);
maxAspectRatio = max([max(aRatio_C1) max(aRatio_C2)]);

figure;
colormap(gray);
subplot(1,2,1);
imagesc(reshape(train_C1_images(50,:,:), 28, 28));

% Compute the rectangle
sum_row = sum(reshape(train_C1_images(50,:,:), 28, 28), 2);
sum_col = sum(reshape(train_C1_images(50,:,:), 28, 28), 1);
min_row = find(sum_row, 1);
max_row = find(sum_row, 1, 'last');
min_col = find(sum_col, 1);
max_col = find(sum_col, 1, 'last');
width = max_col-min_col+2;
height = max_row-min_row+2;
rectangle('Position',[min_col-1 min_row-1 width height], 'EdgeColor','r', 'LineWidth',5)

subplot(1,2,2);
imagesc(reshape(train_C2_images(500,:,:), 28, 28));

% Compute the rectangle
sum_row = sum(reshape(train_C2_images(500,:,:), 28, 28), 2);
sum_col = sum(reshape(train_C2_images(500,:,:), 28, 28), 1);
min_row = find(sum_row, 1);
max_row = find(sum_row, 1, 'last');
min_col = find(sum_col, 1);
max_col = find(sum_col, 1, 'last');
width = max_col-min_col+2;
height = max_row-min_row+2;
rectangle('Position',[min_col-1 min_row-1 width height], 'EdgeColor','r', 'LineWidth',5)
%% Bayesian Classifier

% Prior Probabilities
PC1 = length(train_C1_indices)/(length(train_C1_indices) + length(train_C2_indices));
PC2 = length(train_C2_indices)/(length(train_C1_indices) + length(train_C2_indices));

% Likelihoods
m1 = (1/length(aRatio_C1))*sum(aRatio_C1); 
m2 = (1/length(aRatio_C2))*sum(aRatio_C2);

s1 = sqrt((1/length(aRatio_C1))*sum((aRatio_C1-m1).^2));
s2 = sqrt((1/length(aRatio_C2))*sum((aRatio_C2-m2).^2));

PgivenC1 = 1/(sqrt(2*pi)*s1)*exp(-(aRatio_C1-m1).^2/(2*s1^2));
PgivenC2 = 1/(sqrt(2*pi)*s2)*exp(-(aRatio_C2-m2).^2/(2*s2^2));

% Posterior Probabilities
PC1givenL = PgivenC1*PC1;
PC2givenL = PgivenC2*PC2;

% Classification result
test_aRatio_C1 = zeros(1, size(test_C1_images,1));
BayesClassC1 = zeros(1, size(test_C1_images,1));

for i=1:size(test_C1_images)
    test_aRatio_C1(i) = computeAspectRatio(test_C1_images(i,:,:));
    
    [~,idx1] = min(abs(aRatio_C1-test_aRatio_C1(i)));
    [~,idx2] = min(abs(aRatio_C2-test_aRatio_C1(i)));  
    %idx = find(min(abs(a - n)) == abs(a - n));
    if(PC1givenL(idx1) > PC2givenL(idx2))
        BayesClassC1(i) = 1;
    else
        BayesClassC1(i) = 2;
    end
end

test_aRatio_C2 = zeros(1, size(test_C2_images,1));
BayesClassC2 = zeros(1, size(test_C2_images,1));

for i=1:size(test_C2_images)
    test_aRatio_C2(i) = computeAspectRatio(test_C2_images(i,:,:));
    
    [~,idx1] = min(abs(aRatio_C1-test_aRatio_C2(i))); 
    [~,idx2] = min(abs(aRatio_C2-test_aRatio_C2(i)));  
    %idx = find(min(abs(a - n)) == abs(a - n));
    if(PC1givenL(idx1) < PC2givenL(idx2))
        BayesClassC2(i) = 2;
    else
        BayesClassC2(i) = 1;
    end
end

% Count misclassified digits
count_errors_c1 = sum(BayesClassC1(:)==2);
count_errors_c2 = sum(BayesClassC2(:)==1);

p_error_c1 = count_errors_c1/length(test_C1_indices);
p_error_c2 = count_errors_c2/length(test_C2_indices);

% Total Classification Error (percentage)
Error = (count_errors_c1 + count_errors_c2)/(length(test_C1_indices) + length(test_C2_indices));

disp(['Total Error For Class C1: ' num2str(p_error_c1*100) ' %']);
disp(['Total Error For Class C2: ' num2str(p_error_c2*100) ' %']);
disp(['Total Error(%): ' num2str(Error*100) ' %']);