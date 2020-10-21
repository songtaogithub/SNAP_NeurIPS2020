function [t_vec, x, costs, t] = pgd(x, obj, proj, alpha, its, eps_g)
% implement Perturbed Projected Gradient Descent.
% obj is a problem instance, obj.func, obj.grad, obj.hessian
% proj is the projection operator.
% its= 2000;
%grad = obj.grad(x);
%grad_norm = norm(grad, 2);
%d = size(x, 1);
time = 0;
costs = zeros(1, its);
t_vec = zeros(1,its);
fprintf('Iteration:       ');
%W = WH(1:obj.I, :);
%H = WH(obj.I+1:end, :);
            
for t = 1:its
    fprintf('\b\b\b\b\b\b\b\b%8i',t);
    tStart = tic;
    grad = obj.grad(x);
    g_pi_x = proj(x - grad) - x;
    if t>= 1e5 && norm(g_pi_x(:)) < 1e-38
        break;
    end
    w_new = proj(x(1:obj.I, :) - alpha * grad(1:obj.I, :));
    x(1:obj.I, :) = w_new;
    grad = obj.grad(x);
    h_new = proj(x(obj.I+1:end, :) - alpha * grad(obj.I+1:end, :));
    x(obj.I+1:end, :) = h_new;
       
    costs(t) = obj.func(x);
   % x = x_new;
    tElapsed = toc(tStart);
    time = time + tElapsed;
    t_vec(t) = time;
end
fprintf('\n');
end
