% Example data
data = [2104 399900; 1600 329900; 2400 369000];

% Extract features (X) and output variable (y)
X = data(:, 1);
y = data(:, 2);
m = length(y); % number of training examples

% Feature normalization
X_mean = mean(X);
X_std = std(X);
X = (X - X_mean) ./ X_std;

% Add intercept term to X after normalization
X = [ones(m, 1), X];

% Initialize fitting parameters
theta = zeros(2, 1); % Two parameters including intercept term

% Gradient descent settings
alpha = 0.01;
iterations = 400;

% Run gradient descent, using the function from before
[theta, J_history] = gradientDescent(X, y, theta, alpha, iterations);

% Display the final parameters
fprintf('Theta found by gradient descent:\n');
fprintf('%f\n', theta);

% Plot the convergence graph
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');
title('Cost Function Convergence');

% Predict prices for a feature value (need to normalize the feature first)
feature_normal = ([1650] - X_mean) ./ X_std;
price = [1, feature_normal] * theta; % Remember to include the intercept term

fprintf('Predicted price of a 1650 sq-ft house (using gradient descent):\n $%f\n', price);
