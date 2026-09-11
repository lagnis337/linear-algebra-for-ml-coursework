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


#Gradient descent
xorg=x(:,2);
#normilze the data
sigma = std(x);
mu = mean(x);
x(:,2) = (x(:,2) - mu(2))./ sigma(2);


mse = zeros(100, 100);   % initialize mse to 100x100 matrix of 0's
w0 = linspace(100000,500000,100);
w1 = linspace(1000,300000,100);
for i = 1:length(w0)
	  for j = 1:length(w1)
	  t = [w0(i); w1(j)];
	  mse(i,j) =1/m* (y-x*t)'*(y-x*t);

    end
end

% Plot the surface plot
% Because of the way meshgrids work in the surf command, we need to
% transpose mse before calling surf, or else the axes will be flipped
mse = mse';
figure;
surf(w0, w1, mse);
xlabel('\w0'); ylabel('\w1');
hold on; % keep the surface up so the GD trajectory can be drawn on top of it below


w = zeros(2, 1);
iterations = 100
alpha = 0.07
cost=zeros(iterations,1);
for iter = 1:iterations
%insert your code here
    h = x * w;
    cost(iter) = 1/m * (y - h)' * (y - h);

% this is the code to calculate weights
    grad = -2/m * x' * (y - h);
    w = w - grad * alpha;
%end of your code
     wdraw0(iter)=w(1);
     wdraw1(iter)=w(2);
     plot3(w(1),w(2),cost(iter),'rv','markersize',10)

end
disp("w0g w1g"),disp(w);
% now plot Cost
% technically, the first cost  at the zero-eth iteration
% but Matlab/Octave doesn't have a zero index
figure;
plot(0:iterations-1, cost(1:iterations), '-')
xlabel('Number of iterations')
ylabel('Cost - mean square error ')


#normilize 1650sf
%caling 1650sf your code here:
newxs = (1650 - mu(2)) / sigma(2);
disp("house prediction for 1650s, gradient descent:"),disp([1,newxs]*w);



#plot data and regression line
figure % open a new figure window
# plot your training set (and label the axes):
plot(xorg,y,'o');
ylabel('house cost')
xlabel('house sf')
hold

yp=x*w;
plot(xorg,yp,'-');
