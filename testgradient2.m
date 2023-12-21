% Generate some example data (X, y)
data = [1 2; 2 4; 3 6]; % example data where y = 2 * x
X = [ones(size(data, 1), 1), data(:,1)]; % Add a column of ones to data
y = data(:,2);

% Grid over which we will calculate J
theta0_vals = linspace(-10, 10, 100);
theta1_vals = linspace(-1, 4, 100);

% Initialize J_vals to a matrix of 0's
J_vals = zeros(length(theta0_vals), length(theta1_vals));

% Fill out J_vals
for i = 1:length(theta0_vals)
    for j = 1:length(theta1_vals)
        t = [theta0_vals(i); theta1_vals(j)];
        J_vals(i,j) = computeCost(X, y, t);
    end
end

% Because of the way meshgrids work in the surf command, we need to transpose J_vals
J_vals = J_vals';

% Surface plot
figure;
surf(theta0_vals, theta1_vals, J_vals)
xlabel('\theta_0');
ylabel('\theta_1');
zlabel('Cost J');
title('Cost function surface and contour plot');

hold on;

% Contour plot
contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20), 'LineColor', 'k');

% Assume theta_history is a matrix of theta values over iterations
% This is just example data, replace it with actual theta history from gradient descent
theta_history = [1 0.5; 0.9 0.4; 0.8 0.3; 0.7 0.2]; % Example path of gradient descent
J_history = zeros(1, size(theta_history, 1));

% Calculate the cost function for each pair of theta in the history
for i = 1:size(theta_history, 1)
    J_history(i) = computeCost(X, y, theta_history(i,:)');
end

% Plot the path of gradient descent
plot3(theta_history(:,1), theta_history(:,2), J_history, 'w-', 'LineWidth', 2);

% Adjust the viewing angle for better visibility
view(-60, 10);

hold off;


% New figure for the 2D plot
figure;

% Contour plot
% Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 2, 15))
xlabel('\theta_0');
ylabel('\theta_1');
title('Contour plot of the cost function J with Gradient Descent Path');
hold on;

% Overlay the path of gradient descent
plot(theta_history(:,1), theta_history(:,2), 'r-', 'LineWidth', 2);
plot(theta_history(:,1), theta_history(:,2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);

% Annotate the start and end points
text(theta_history(1,1), theta_history(1,2), 'Start', 'VerticalAlignment', 'Top', 'HorizontalAlignment', 'Left');
text(theta_history(end,1), theta_history(end,2), 'End', 'VerticalAlignment', 'Bottom', 'HorizontalAlignment', 'Right');

hold off;