clear all;
close all;
clc;

load('digits.mat');
A = reshape(train7(43, :), 28, 28)';
imagesc(A);

n = size(train0,1);
m = size(train0,2);
num_classes = 10;

priors = zeros(num_classes,m);

priors(1,:) = (1/n)*sum(train0);
priors(2,:) = (1/n)*sum(train1);
priors(3,:) = (1/n)*sum(train2);
priors(4,:) = (1/n)*sum(train3);
priors(5,:) = (1/n)*sum(train4);
priors(6,:) = (1/n)*sum(train5);
priors(7,:) = (1/n)*sum(train6);
priors(8,:) = (1/n)*sum(train7);
priors(9,:) = (1/n)*sum(train8);
priors(10,:) = (1/n)*sum(train9);

priors(priors==0) = 0.0000001;

figure;
suptitle('Trained Models of every digit');
temp = zeros(n,1);
for i=1:num_classes
    for j=1:n
        if i==1 
            temp(j) = train0(j,:)*log(priors(i,:))' + (1-train0(j,:))*log(1-priors(i,:))';
        elseif i==2 
            temp(j) = train1(j,:)*log(priors(i,:))' + (1-train1(j,:))*log(1-priors(i,:))';
        elseif i==3 
            temp(j) = train2(j,:)*log(priors(i,:))' + (1-train2(j,:))*log(1-priors(i,:))';
        elseif i==4 
            temp(j) = train3(j,:)*log(priors(i,:))' + (1-train3(j,:))*log(1-priors(i,:))';
        elseif i==5 
            temp(j) = train4(j,:)*log(priors(i,:))' + (1-train4(j,:))*log(1-priors(i,:))';
        elseif i==6 
            temp(j) = train5(j,:)*log(priors(i,:))' + (1-train5(j,:))*log(1-priors(i,:))';
        elseif i==7 
            temp(j) = train6(j,:)*log(priors(i,:))' + (1-train6(j,:))*log(1-priors(i,:))';
        elseif i==8 
            temp(j) = train7(j,:)*log(priors(i,:))' + (1-train7(j,:))*log(1-priors(i,:))';
        elseif i==9 
            temp(j) = train8(j,:)*log(priors(i,:))' + (1-train8(j,:))*log(1-priors(i,:))';
        elseif i==10 
            temp(j) = train9(j,:)*log(priors(i,:))' + (1-train9(j,:))*log(1-priors(i,:))';
        end
    end
    if i==1 
        image = train0'*temp;
    elseif i==2 
        image = train1'*temp;
    elseif i==3 
        image = train2'*temp;
    elseif i==4 
        image = train3'*temp;
    elseif i==5 
        image = train4'*temp;
    elseif i==6 
        image = train5'*temp;
    elseif i==7 
        image = train6'*temp;
    elseif i==8 
        image = train7'*temp;
    elseif i==9 
        image = train8'*temp;
    elseif i==10 
        image = train9'*temp;
    end
    
    subplot(2,5,i);
    imagesc(reshape(image, 28, 28)');
end


confusion_matrix = zeros(10,10);
likelihoods = zeros(1,num_classes);
digit_accuracy = zeros(1,num_classes);

digit_counter = 0;          % Count the correct classified digits of every class
classifier_counter = 0;     % Count the total correct classified digits

for i=1:num_classes
  digit_counter = 0;
  for j=1:n
    if i==1
       likelihoods(1) = test0(j,:)*log(priors(1,:))' + (1-test0(j,:))*log(1-priors(1,:))';
       likelihoods(2) = test0(j,:)*log(priors(2,:))' + (1-test0(j,:))*log(1-priors(2,:))';
       likelihoods(3) = test0(j,:)*log(priors(3,:))' + (1-test0(j,:))*log(1-priors(3,:))';
       likelihoods(4) = test0(j,:)*log(priors(4,:))' + (1-test0(j,:))*log(1-priors(4,:))';
       likelihoods(5) = test0(j,:)*log(priors(5,:))' + (1-test0(j,:))*log(1-priors(5,:))';
       likelihoods(6) = test0(j,:)*log(priors(6,:))' + (1-test0(j,:))*log(1-priors(6,:))';
       likelihoods(7) = test0(j,:)*log(priors(7,:))' + (1-test0(j,:))*log(1-priors(7,:))';
       likelihoods(8) = test0(j,:)*log(priors(8,:))' + (1-test0(j,:))*log(1-priors(8,:))';
       likelihoods(9) = test0(j,:)*log(priors(9,:))' + (1-test0(j,:))*log(1-priors(9,:))';
       likelihoods(10) = test0(j,:)*log(priors(10,:))' + (1-test0(j,:))*log(1-priors(10,:))';
       
    elseif i==2 
       likelihoods(1) = test1(j,:)*log(priors(1,:))' + (1-test1(j,:))*log(1-priors(1,:))';
       likelihoods(2) = test1(j,:)*log(priors(2,:))' + (1-test1(j,:))*log(1-priors(2,:))';
       likelihoods(3) = test1(j,:)*log(priors(3,:))' + (1-test1(j,:))*log(1-priors(3,:))';
       likelihoods(4) = test1(j,:)*log(priors(4,:))' + (1-test1(j,:))*log(1-priors(4,:))';
       likelihoods(5) = test1(j,:)*log(priors(5,:))' + (1-test1(j,:))*log(1-priors(5,:))';
       likelihoods(6) = test1(j,:)*log(priors(6,:))' + (1-test1(j,:))*log(1-priors(6,:))';
       likelihoods(7) = test1(j,:)*log(priors(7,:))' + (1-test1(j,:))*log(1-priors(7,:))';
       likelihoods(8) = test1(j,:)*log(priors(8,:))' + (1-test1(j,:))*log(1-priors(8,:))';
       likelihoods(9) = test1(j,:)*log(priors(9,:))' + (1-test1(j,:))*log(1-priors(9,:))';
       likelihoods(10) = test1(j,:)*log(priors(10,:))' + (1-test1(j,:))*log(1-priors(10,:))';
       
    elseif i==3 
       likelihoods(1) = test2(j,:)*log(priors(1,:))' + (1-test2(j,:))*log(1-priors(1,:))';
       likelihoods(2) = test2(j,:)*log(priors(2,:))' + (1-test2(j,:))*log(1-priors(2,:))';
       likelihoods(3) = test2(j,:)*log(priors(3,:))' + (1-test2(j,:))*log(1-priors(3,:))';
       likelihoods(4) = test2(j,:)*log(priors(4,:))' + (1-test2(j,:))*log(1-priors(4,:))';
       likelihoods(5) = test2(j,:)*log(priors(5,:))' + (1-test2(j,:))*log(1-priors(5,:))';
       likelihoods(6) = test2(j,:)*log(priors(6,:))' + (1-test2(j,:))*log(1-priors(6,:))';
       likelihoods(7) = test2(j,:)*log(priors(7,:))' + (1-test2(j,:))*log(1-priors(7,:))';
       likelihoods(8) = test2(j,:)*log(priors(8,:))' + (1-test2(j,:))*log(1-priors(8,:))';
       likelihoods(9) = test2(j,:)*log(priors(9,:))' + (1-test2(j,:))*log(1-priors(9,:))';
       likelihoods(10) = test2(j,:)*log(priors(10,:))' + (1-test2(j,:))*log(1-priors(10,:))';
       
    elseif i==4 
       likelihoods(1) = test3(j,:)*log(priors(1,:))' + (1-test3(j,:))*log(1-priors(1,:))';
       likelihoods(2) = test3(j,:)*log(priors(2,:))' + (1-test3(j,:))*log(1-priors(2,:))';
       likelihoods(3) = test3(j,:)*log(priors(3,:))' + (1-test3(j,:))*log(1-priors(3,:))';
       likelihoods(4) = test3(j,:)*log(priors(4,:))' + (1-test3(j,:))*log(1-priors(4,:))';
       likelihoods(5) = test3(j,:)*log(priors(5,:))' + (1-test3(j,:))*log(1-priors(5,:))';
       likelihoods(6) = test3(j,:)*log(priors(6,:))' + (1-test3(j,:))*log(1-priors(6,:))';
       likelihoods(7) = test3(j,:)*log(priors(7,:))' + (1-test3(j,:))*log(1-priors(7,:))';
       likelihoods(8) = test3(j,:)*log(priors(8,:))' + (1-test3(j,:))*log(1-priors(8,:))';
       likelihoods(9) = test3(j,:)*log(priors(9,:))' + (1-test3(j,:))*log(1-priors(9,:))';
       likelihoods(10) = test3(j,:)*log(priors(10,:))' + (1-test3(j,:))*log(1-priors(10,:))';

    elseif i==5 
       likelihoods(1) = test4(j,:)*log(priors(1,:))' + (1-test4(j,:))*log(1-priors(1,:))';
       likelihoods(2) = test4(j,:)*log(priors(2,:))' + (1-test4(j,:))*log(1-priors(2,:))';
       likelihoods(3) = test4(j,:)*log(priors(3,:))' + (1-test4(j,:))*log(1-priors(3,:))';
       likelihoods(4) = test4(j,:)*log(priors(4,:))' + (1-test4(j,:))*log(1-priors(4,:))';
       likelihoods(5) = test4(j,:)*log(priors(5,:))' + (1-test4(j,:))*log(1-priors(5,:))';
       likelihoods(6) = test4(j,:)*log(priors(6,:))' + (1-test4(j,:))*log(1-priors(6,:))';
       likelihoods(7) = test4(j,:)*log(priors(7,:))' + (1-test4(j,:))*log(1-priors(7,:))';
       likelihoods(8) = test4(j,:)*log(priors(8,:))' + (1-test4(j,:))*log(1-priors(8,:))';
       likelihoods(9) = test4(j,:)*log(priors(9,:))' + (1-test4(j,:))*log(1-priors(9,:))';
       likelihoods(10) = test4(j,:)*log(priors(10,:))' + (1-test4(j,:))*log(1-priors(10,:))';
            
    elseif i==6 
       likelihoods(1) = test5(j,:)*log(priors(1,:))' + (1-test5(j,:))*log(1-priors(1,:))';
       likelihoods(2) = test5(j,:)*log(priors(2,:))' + (1-test5(j,:))*log(1-priors(2,:))';
       likelihoods(3) = test5(j,:)*log(priors(3,:))' + (1-test5(j,:))*log(1-priors(3,:))';
       likelihoods(4) = test5(j,:)*log(priors(4,:))' + (1-test5(j,:))*log(1-priors(4,:))';
       likelihoods(5) = test5(j,:)*log(priors(5,:))' + (1-test5(j,:))*log(1-priors(5,:))';
       likelihoods(6) = test5(j,:)*log(priors(6,:))' + (1-test5(j,:))*log(1-priors(6,:))';
       likelihoods(7) = test5(j,:)*log(priors(7,:))' + (1-test5(j,:))*log(1-priors(7,:))';
       likelihoods(8) = test5(j,:)*log(priors(8,:))' + (1-test5(j,:))*log(1-priors(8,:))';
       likelihoods(9) = test5(j,:)*log(priors(9,:))' + (1-test5(j,:))*log(1-priors(9,:))';
       likelihoods(10) = test5(j,:)*log(priors(10,:))' + (1-test5(j,:))*log(1-priors(10,:))';

    elseif i==7 
       likelihoods(1) = test6(j,:)*log(priors(1,:))' + (1-test6(j,:))*log(1-priors(1,:))';
       likelihoods(2) = test6(j,:)*log(priors(2,:))' + (1-test6(j,:))*log(1-priors(2,:))';
       likelihoods(3) = test6(j,:)*log(priors(3,:))' + (1-test6(j,:))*log(1-priors(3,:))';
       likelihoods(4) = test6(j,:)*log(priors(4,:))' + (1-test6(j,:))*log(1-priors(4,:))';
       likelihoods(5) = test6(j,:)*log(priors(5,:))' + (1-test6(j,:))*log(1-priors(5,:))';
       likelihoods(6) = test6(j,:)*log(priors(6,:))' + (1-test6(j,:))*log(1-priors(6,:))';
       likelihoods(7) = test6(j,:)*log(priors(7,:))' + (1-test6(j,:))*log(1-priors(7,:))';
       likelihoods(8) = test6(j,:)*log(priors(8,:))' + (1-test6(j,:))*log(1-priors(8,:))';
       likelihoods(9) = test6(j,:)*log(priors(9,:))' + (1-test6(j,:))*log(1-priors(9,:))';
       likelihoods(10) = test6(j,:)*log(priors(10,:))' + (1-test6(j,:))*log(1-priors(10,:))';

    elseif i==8 
       likelihoods(1) = test7(j,:)*log(priors(1,:))' + (1-test7(j,:))*log(1-priors(1,:))';
       likelihoods(2) = test7(j,:)*log(priors(2,:))' + (1-test7(j,:))*log(1-priors(2,:))';
       likelihoods(3) = test7(j,:)*log(priors(3,:))' + (1-test7(j,:))*log(1-priors(3,:))';
       likelihoods(4) = test7(j,:)*log(priors(4,:))' + (1-test7(j,:))*log(1-priors(4,:))';
       likelihoods(5) = test7(j,:)*log(priors(5,:))' + (1-test7(j,:))*log(1-priors(5,:))';
       likelihoods(6) = test7(j,:)*log(priors(6,:))' + (1-test7(j,:))*log(1-priors(6,:))';
       likelihoods(7) = test7(j,:)*log(priors(7,:))' + (1-test7(j,:))*log(1-priors(7,:))';
       likelihoods(8) = test7(j,:)*log(priors(8,:))' + (1-test7(j,:))*log(1-priors(8,:))';
       likelihoods(9) = test7(j,:)*log(priors(9,:))' + (1-test7(j,:))*log(1-priors(9,:))';
       likelihoods(10) = test7(j,:)*log(priors(10,:))' + (1-test7(j,:))*log(1-priors(10,:))';

    elseif i==9 
       likelihoods(1) = test8(j,:)*log(priors(1,:))' + (1-test8(j,:))*log(1-priors(1,:))';
       likelihoods(2) = test8(j,:)*log(priors(2,:))' + (1-test8(j,:))*log(1-priors(2,:))';
       likelihoods(3) = test8(j,:)*log(priors(3,:))' + (1-test8(j,:))*log(1-priors(3,:))';
       likelihoods(4) = test8(j,:)*log(priors(4,:))' + (1-test8(j,:))*log(1-priors(4,:))';
       likelihoods(5) = test8(j,:)*log(priors(5,:))' + (1-test8(j,:))*log(1-priors(5,:))';
       likelihoods(6) = test8(j,:)*log(priors(6,:))' + (1-test8(j,:))*log(1-priors(6,:))';
       likelihoods(7) = test8(j,:)*log(priors(7,:))' + (1-test8(j,:))*log(1-priors(7,:))';
       likelihoods(8) = test8(j,:)*log(priors(8,:))' + (1-test8(j,:))*log(1-priors(8,:))';
       likelihoods(9) = test8(j,:)*log(priors(9,:))' + (1-test8(j,:))*log(1-priors(9,:))';
       likelihoods(10) = test8(j,:)*log(priors(10,:))' + (1-test8(j,:))*log(1-priors(10,:))';

    elseif i==10 
       likelihoods(1) = test9(j,:)*log(priors(1,:))' + (1-test9(j,:))*log(1-priors(1,:))';
       likelihoods(2) = test9(j,:)*log(priors(2,:))' + (1-test9(j,:))*log(1-priors(2,:))';
       likelihoods(3) = test9(j,:)*log(priors(3,:))' + (1-test9(j,:))*log(1-priors(3,:))';
       likelihoods(4) = test9(j,:)*log(priors(4,:))' + (1-test9(j,:))*log(1-priors(4,:))';
       likelihoods(5) = test9(j,:)*log(priors(5,:))' + (1-test9(j,:))*log(1-priors(5,:))';
       likelihoods(6) = test9(j,:)*log(priors(6,:))' + (1-test9(j,:))*log(1-priors(6,:))';
       likelihoods(7) = test9(j,:)*log(priors(7,:))' + (1-test9(j,:))*log(1-priors(7,:))';
       likelihoods(8) = test9(j,:)*log(priors(8,:))' + (1-test9(j,:))*log(1-priors(8,:))';
       likelihoods(9) = test9(j,:)*log(priors(9,:))' + (1-test9(j,:))*log(1-priors(9,:))';
       likelihoods(10) = test9(j,:)*log(priors(10,:))' + (1-test9(j,:))*log(1-priors(10,:))';
            
    end
    
    [~,index] = max(likelihoods);
    confusion_matrix(i,index) = confusion_matrix(i,index) + 1;
    
    if(i==index)
      digit_counter = digit_counter + 1;
      classifier_counter = classifier_counter + 1;
    end
  end
  digit_accuracy(i) = digit_counter/n;
  disp(['Accuracy for digit ',num2str(i-1),' is: ',num2str(digit_accuracy(i))]);
end

classifier_accuracy = classifier_counter/(n*num_classes);
disp(' ');
disp(['Total Accuracy of Naive Bayesian classifier is: ', num2str(classifier_accuracy)]);
confusion_matrix = confusion_matrix/n;

disp(' ');
disp('================================================Confussion Matrix================================================');
disp(num2str(confusion_matrix));
disp('=================================================================================================================');