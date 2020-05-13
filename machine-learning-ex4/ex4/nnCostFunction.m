function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

X = [ones(m,1) X];



ybin=zeros(m,num_labels);
for iter=1:m
    ybin(iter,y(iter,1)) = 1;
end

%%
% for t=1:m
%     
%     a1= X(t,:)';            % linha 1x401
%     z2= a1*(Theta1');       % linha 1x25
%     a2= sigmoid(z2);
%     a2= [1 a2];             % linha 1x26
%     z3= a2*(Theta2');       % linha 1x10
%     a3= sigmoid(z3);
%     
%     d3= a3 - ybin;          % linha 1x10
%     
%     d2= d3*(Theta2) .*a2;
%     
% end
%%

% D1=0; D2=0;
% 
% for t=1:m
%     
%     %1
%     a1= X(t,:);    % 1x4
%     z2= a1*(Theta1');    % 1x4. 4x5 = 1x5
%     a2= sigmoid(z2);
%     a2= [1 a2];    % 1x6
%     z3= a2*(Theta2');   % 1x6. 6x3 = 1x3
%     a3= sigmoid(z3);
%     
%     % ybin = zeros(1,num_labels);
%     % ybin(1,y) = 1; % vec.linha binário para um alvo
%     
%     %2
%     ybin1 = ybin(t,:); % 1x3
%     d3= a3 - ybin1; % 1x3
%     
%     %3
% %     size((Theta2)* (d3)), size(sigmoidGradient(z2))
%     d2= (Theta2(:,2:end)')* (d3');   % 5x3. 3x1 = 5x1
%     d2= d2'.* sigmoidGradient(z2);  % 1x5 .* 1x5 = 1x5
%     
%     %4
%     d2= d2(1,2:end);    %1x4
%     
% %     size(d2), size(a1')
%     D1= D1 + d2*(a1');
% %     size(d3), size(a2)
%     D2= D2 + d3*(a2(1,4:end)');
%     
%     Theta1_grad= D1;
%     Theta2_grad= D2;
% end
% 
% Theta1_grad= (1/m)*D1;
% Theta2_grad= (1/m)*D2;




a1= X;
z2= a1*(Theta1');
a2= sigmoid(z2);
a2= [ones(m,1) a2];
z3= a2*(Theta2');
a3= sigmoid(z3);

for i=1:1:m
    for k=1:1:num_labels
        J= J -1*ybin(i,k)*log(a3(i,k)) -1*(1-ybin(i,k))*log( 1-a3(i,k) );
    end
end

J = (1/m)*J;

% REGULARIZATION

t1=0; t2=0;

for j=1:1:size(Theta1,1)
    for k=2:1:size(Theta1,2)
        t1= t1 + (Theta1(j,k))^2;
    end
end

for j=1:1:size(Theta2,1)
    for k=2:1:size(Theta2,2)
        t2= t2 + (Theta2(j,k))^2;
    end
end

reg= (lambda/(2*m))*(t1+t2);

J= J + reg;

% BACK

d3= a3 - ybin;
disp('d3='), size(d3)
disp('Theta2='), size(Theta2)

% size(d3),size(Theta2),size(a2)
d21= d3*(Theta2(:,2:end));
d2= d21 .*( sigmoid(z2).*(1-sigmoid(z2)) );
disp('d2='), size(d2)

% size(d2)
% d2= d2(:,2:end);

% size(d2),size(a1)
Delta1= d2'*a1;
disp('DELTA1='), size(Delta1)

% size(d3), size(a2)
Delta2= d3'*a2;
disp('DELTA2='), size(Delta2)

Theta1_grad= (1/m)*Delta1;
Theta2_grad= (1/m)*Delta2;
disp('gradT1='), size(Theta1_grad)
disp('gradT2='), size(Theta2_grad)

% REGUL

Theta1(:,1)=zeros(size(Theta1,1),1);
Theta2(:,1)=zeros(size(Theta2,1),1);

Theta1= (lambda/m)*Theta1;
Theta2= (lambda/m)*Theta2;

Theta1_grad = Theta1_grad + Theta1;
Theta2_grad = Theta2_grad + Theta2;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
