clear all;
close all;
clc;

S1 = [1.2 -0.4; -0.4 1.2];
S2 = [1.2 0.4; 0.4 1.2];

mu1 = [3 3];
mu2 = [6 6];

x1 = -3:0.01:15;
x2 = -3:0.01:15;

[X1,X2] = meshgrid(x1,x2);
figure(1);
hold on
Y1 = mvnpdf([X1(:) X2(:)],mu1,S1);
Y1 = reshape(Y1,length(x2),length(x1));
contour(x1,x2,Y1,[0.0001 0.001 0.01 0.05 0.15 0.25 0.35]);
hold off;

hold on;
Y2 = mvnpdf([X1(:) X2(:)],mu2,S2);
Y2 = reshape(Y2,length(x2),length(x1));
contour(x1,x2,Y2,[0.0001 0.001 0.01 0.05 0.15 0.25 0.35]);
hold off;

% B)
Pw1 = [0.1 0.25 0.5 0.75 0.9];
Pw2 = 1-Pw1 ;

syms x y real
hold on;
for i=1:length(Pw1)
    eq1 = Pw1(i)*exp(-(1/2)*[x-mu1(1) y-mu1(2)]*(1/det(S1))*inv(S1)*[x-mu1(1) y-mu1(2)]');
    eq2 = Pw2(i)*exp(-(1/2)*[x-mu2(1) y-mu2(2)]*(1/det(S2))*inv(S2)*[x-mu2(1) y-mu2(2)]');

    eq = eq1==eq2;
    y_out = solve(eq,y);

    fplot(y_out);
end
hold off;
grid on
axis([-3 16 -3 16]);
legend('Ισουψείς κλάσης ω1','Ισουψείς κλάσεις ω2','Pω1=0.1','Pω1=0.25','Pω1=0.5','Pω1=0.75','Pω1=0.9');

%% 
S1 = [1.2 0.4; 0.4 1.2];
S2 = [1.2 0.4; 0.4 1.2];

mu1 = [3 3];
mu2 = [6 6];

x1 = -3:0.01:15;
x2 = -3:0.01:15;

[X1,X2] = meshgrid(x1,x2);
figure(2);
hold on
Y1 = mvnpdf([X1(:) X2(:)],mu1,S1);
Y1 = reshape(Y1,length(x2),length(x1));
contour(x1,x2,Y1,[0.0001 0.001 0.01 0.05 0.15 0.25 0.35]);
hold off;

hold on;
Y2 = mvnpdf([X1(:) X2(:)],mu2,S2);
Y2 = reshape(Y2,length(x2),length(x1));
contour(x1,x2,Y2,[0.0001 0.001 0.01 0.05 0.15 0.25 0.35]);
hold off;

% B)
Pw1 = [0.1 0.25 0.5 0.75 0.9];
Pw2 = 1-Pw1 ;

syms x y real
hold on;
for i=1:length(Pw1)
    eq1 = Pw1(i)*exp(-(1/2)*[x-mu1(1) y-mu1(2)]*(1/det(S1))*inv(S1)*[x-mu1(1) y-mu1(2)]');
    eq2 = Pw2(i)*exp(-(1/2)*[x-mu2(1) y-mu2(2)]*(1/det(S2))*inv(S2)*[x-mu2(1) y-mu2(2)]');
    
    eq = eq1==eq2;
    y_out = solve(eq,y);

    fplot(y_out);
end
hold off;
grid on
axis([-3 16 -3 16]);
legend('Ισουψείς κλάσης ω1','Ισουψείς κλάσεις ω2','Pω1=0.1','Pω1=0.25','Pω1=0.5','Pω1=0.75','Pω1=0.9');

