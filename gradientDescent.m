function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
    % Initialize some values
    m = length(y); % number of training examples
    J_history = zeros(num_iters, 1);

    for iter = 1:num_iters
        % Calculate the hypothesis
        h = X * theta;
        
        % Errors
        errors = h - y;
        
        % Update theta by taking a step with size alpha in the direction of the negative gradient
        theta = theta - (alpha/m) * (X' * errors);
        
        % Save the cost J in every iteration    
        J_history(iter) = (1/(2*m)) * sum(errors .^ 2);
    end
end