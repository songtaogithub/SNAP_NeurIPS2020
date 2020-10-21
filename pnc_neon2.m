function [t_vec, costs, count_eig] = pnc_neon2(x, obj, proj, eps_g, alpha, its)
% implement Perturbed Projected Gradient Descent.
% obj is a problem instance, obj.func, obj.grad, obj.hessian
% proj is the projection operator.
costs = zeros(1, its);
count_eig = 0;
fprintf('Iteration:        ');
% llratio = 1;
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
        
        act_set_x = ~logical(x); % TODO: need to be modified for other const
        %               Hess = obj.hessian(x);
        %               Hess(act_set_x, :) = [];
        %               Hess(:, act_set_x) = [];
        %         [U, D] = eig(Hess);
        %         [d, ind] = sort(diag(D));
        %         u1 = U(:, ind(1));
        
        [u1] = spneon2(x, obj, act_set_x, grad);
        dr = zeros(size(x));
        dr(~act_set_x) = u1(~act_set_x);
        q_pi_x = grad;
        q_pi_x(act_set_x) = 0;
        if dr(:)'* q_pi_x(:) > 0
            dr = -dr;
        end
        [x_new_nc, alpha_flag_nc] = line_search(obj, x, dr, 0.01, false, act_thres);
        [x_new_gd, alpha_flag_gd] = line_search(obj, x, -q_pi_x, 0.01, true, act_thres);
        % x_new_gd = proj(x - alpha * grad);
        if obj.func(x_new_nc)-obj.func(x) < (obj.func(x_new_gd)-obj.func(x))
            x_new = x_new_nc;
            count_eig = count_eig + 1;
            t_last = t;
            alpha_flag = alpha_flag_nc;
            stopc = false;
        else
            x_new = x_new_gd;
            alpha_flag = 1;
            stopc = true;
        end
        
        if norm(g_pi_x(:)) < 1e-20  && t>=1e4  &&  stopc
            %if norm(g_pi_x) < 1e-12  &&  nc_flag == 0 && t>=1e4
            break;
        else
            x_new = proj(x_new - alpha * obj.grad(x_new));
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

