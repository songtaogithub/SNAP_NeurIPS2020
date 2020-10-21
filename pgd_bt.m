function [t_vec, x, costs, t, alphas, grads] = pgd_bt(x, obj, proj, alpha, its, eps_g)
% implement Perturbed Projected Gradient Descent, using simple backtracking
% obj is a problem instance, obj.func, obj.grad, obj.hessian
% proj is the projection operator.
% its= 2000;
grad = obj.grad(x);
d = size(x, 1);
costs = zeros(1, its);
alphas = zeros(its, 1);
grads = zeros(1, its);
fprintf('Iteration:          ');
time = 0;
t_vec = zeros(1,its);
% t_int = 200;
% t_last = 0;
% alpha_c = alpha;
for t = 1:its
    fprintf('\b\b\b\b\b\b\b\b%8i',t);
    tStart = tic;
    grad = obj.grad(x);
    g_pi_x = proj(x - grad) - x;
    if t >= 3e4 && norm(g_pi_x) < 1e-12
        break;
    end
    %     if t - t_last >= t_int
    %         t_last = t;
    alpha = bt(x, obj, -grad);
    alphas(t) = alpha;
    x_new = proj(x - alpha * grad);
    %     else
    %         x_new = proj(x - alpha_c * grad);
    
    %     end
    costs(t) = obj.func(x_new);
    grads(t) = norm(proj(x_new - grad) - x_new, 'fro');
    x = x_new;
    tElapsed = toc(tStart);
    time = time + tElapsed;
    t_vec(t) = time;
end
fprintf('\n');
end
