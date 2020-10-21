function [t_vec, costs, count_eig] = ppgd_eig(x, obj, proj, eps_g, alpha, its)
% implement Perturbed Projected Gradient Descent.
% obj is a problem instance, obj.func, obj.grad, obj.hessian
% proj is the projection operator.
costs = zeros(1, its);
count_eig = 0;
fprintf('Iteration:        ');
llratio = 0.1;
t_int = 600;
t_last = 0;
act_thres = 0; % values below this considered as 0
%eps_g_t = eps_g*1e5;
alpha_flag = 1;
time = 0;
t_vec = zeros(1,its);
for t = 1:its
    fprintf('\b\b\b\b\b\b\b\b%8i',t);
    tStart = tic;
    grad = obj.grad(x);
    g_pi_x = proj(x - grad) - x;
    if norm(g_pi_x(:)) < eps_g && ( alpha_flag == 1 || (alpha_flag == 0 && t - t_last >= t_int))
        t_last = t;
        act_set_x = ~logical(x); % TODO: need to be modified for other const
        Hess = obj.hessian(x);
        Hess(act_set_x, :) = [];
        Hess(:, act_set_x) = [];
        [U, D] = eig(Hess);
        [d, ind] = sort(diag(D));
        u1 = U(:, ind(1));
        if norm(g_pi_x(:)) < 1e-20  &&  d(1) > -0.01
            break;
        end
        if d(1) <= 0
            dr = zeros(size(x));
            dr(~act_set_x) = u1;
            q_pi_x = grad;
            q_pi_x(act_set_x) = 0;
            if dr(:)'* q_pi_x(:) > 0
                dr = -dr;
            end
            if abs(d(1)) * dr(:)' * q_pi_x(:) / 2 - abs(d(1))^3/ 2 < - llratio*norm(q_pi_x(:))^2 %&& d(1)<0
                count_eig = count_eig + 1;
                is_grad = false;
            else
                dr = -q_pi_x;
                is_grad = true;
            end
            [x_new, alpha_flag] = line_search(obj, x, dr, abs(d(1)), is_grad, act_thres);
        else
            x_new = proj(x - alpha * grad);
        end
    else
        x_new = proj(x - alpha * grad);
    end
    costs(t) = obj.func(x_new);
    x = x_new;
    tElapsed = toc(tStart);
    time = time + tElapsed;
    t_vec(t) = time;
end
fprintf('\n');
end

