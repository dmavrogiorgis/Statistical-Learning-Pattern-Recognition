clear all;
close all;
clc;

x1 = [2, 2, -2, -2]';
y1 = [2, -2, -2, 2]';
w1(:,2:3) = [x1,y1];
w1(:,1) = 1;

x2 = [1, 1, -1, -1]';
y2 = [1, -1, -1, 1]';
w2(:,2:3) = [x2,y2];
w2(:,1) = -1;

twofeature1 = [w1;w2];

n = size(twofeature1,1); % leave out the last example
y = twofeature1(1:n, 1);
X = twofeature1(1:n, 2:3);

Xpos = X(y==1,:); % positive examples
Xneg = X(y==-1,:); % negative examples

%  Visualize the example dataset
figure;
subplot(1,2,1);
hold on;
plot(Xpos(:,1), Xpos(:,2), 'b.');
plot(Xneg(:,1), Xneg(:,2), 'r.');
hold off;
grid on;
axis([-3 3, -3 3]);
title('Initial Dataset');

subplot(1,2,2);
X = X - X(:,1).^2 - X(:,2).^2 - 4;

Xpos = X(y==1,:); % positive examples
Xneg = X(y==-1,:); % negative examples

hold on;
plot(Xpos(:,1), Xpos(:,2), 'b.');
plot(Xneg(:,1), Xneg(:,2), 'r.');
hold off;
grid on;
axis([-15 -4, -15 -4]);

% Form the matrices for the quadratic optimization. See the matlab manual for "quadprog"
H = (X*X').*(y*y');

f = -ones(n,1);

A = [];

b = [];

Aeq = y';

beq = 0; 

lb = zeros(n,1);

ub = Inf*ones(n,1);

lambda = quadprog(H, f, A, b, Aeq, beq, lb, ub); % Find the Lagrange multipliers

indices = find(lambda > 0.0001); % Find the support vectors
Xsup = X(indices,:); % The support vectors only 
ysup = y(indices,:);
lambdasup = lambda(indices);

% Find the weights
w = (lambda.*y)'*X;

% Find the bias term w0
w0 = mean(ysup - Xsup*w');

% Plot support vectors
Xsup_pos = Xsup(ysup==1,:);
Xsup_neg = Xsup(ysup==-1,:);

hold on;
plot(Xsup_pos(:,1), Xsup_pos(:,2), 'bo');
plot(Xsup_neg(:,1), Xsup_neg(:,2), 'ro');
hold off;


% Find the width of the margin
width = -w0/max(w);

% Plot decision boundary
x = linspace(-15, -4, 100);
y = -(w(1)/w(2))*x + width;
hold on;
plot(x, y, 'k');
plot(x, y+(1/max(w)), 'b');  %Plot the Margin of class +1
plot(x, y-(1/max(w)), 'r');  %Plot the Margin of class -1
hold off;
title('Ö(x) transform to dataset and decision boundary');