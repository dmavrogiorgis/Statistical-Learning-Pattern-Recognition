clear all;
close all;
clc;

x1 = [3, 3, 5, 5]';
y1 = [1, -1, 1, -1]';
w1(:,2:3) = [x1,y1];
w1(:,1) = 1;

x2 = [1, 0, 0, -1]';
y2 = [0, 1, -1, 0]';
w2(:,2:3) = [x2,y2];
w2(:,1) = -1;

twofeature1 = [w1;w2];

n = size(twofeature1,1); % leave out the last example
y = twofeature1(1:n, 1);
X = twofeature1(1:n, 2:3);

Xpos = X(y==1,:); % positive examples
Xneg = X(y==-1,:); % negative examples

%  Visualize the example dataset
hold on;
plot(Xpos(:,1), Xpos(:,2), 'b.');
plot(Xneg(:,1), Xneg(:,2), 'r.');
hold off;
axis square;

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
x = linspace(-2, 2, 100);
y = -(w(2)/w(1))*x + width;
hold on;
plot(y, x, 'k');
plot(y+(1/max(w)), x, 'b');  %Plot the Margin of class +1
plot(y-(1/max(w)), x, 'r');  %Plot the Margin of class -1
hold off;
grid on;
axis([-2 6 -2 2]);
title('Initial Dataset and decision boundary');