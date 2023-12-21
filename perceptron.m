function output = perceptron(input, weights, threshold)
    % Calculate the weighted sum
    weighted_sum = dot(input, weights);
    
    % Apply the activation function (step function)
    if weighted_sum >= threshold
        output = 1;
    else
        output = 0;
    end
end
