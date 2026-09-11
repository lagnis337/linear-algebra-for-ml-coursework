%in this exercise, you will investigate linear regression using gradient descent
% and the normal equations
%This is a training set of housing prices in Portland, Oregon, where
%y are the prices and the inputs $x are the living area and the number of bedrooms.
%we will use only living area in this assigment
% initialize samples of x and y training dataset
%
x2 = load('ex3x.txt');
y = load('ex3y.txt');  %house cost
x=x2(:,1);  %only living area

m = length(y); % store the number of training examples
x = [ones(m, 1), x]; % Add a column of ones to x
%the area values from your training data are actually in the second column of x


%calculate parametar w using normal equation
 %your code here
 %
 wn = inv(x' * x) * x' * y;
 %code end
disp("w0 w1"),disp(wn);

#predict the house price with 1650sf
disp("house prediction for 1650s, normal eq:"),disp([1,1650]*wn);
yp=x*wn;

#plot data and regression line
figure % open a new figure window
# plot your training set (and label the axes):
plot(x(:,2),y,'o');
ylabel('house cost')
xlabel('house sf')
hold
plot(x(:,2),yp,'-');
