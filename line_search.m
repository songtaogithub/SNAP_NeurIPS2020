function [x_r1, flag] = line_search(obj, x_r, d_r, eps_h, is_grad, act_thres)
% func is a function handle to evaluate objective function
% const_oracle is a function handle to test if a given point is feasible
% x_r, d_r: current iterate, search direction.
% upper: upperbound constraints
%d = size(x_r(:));
ar_max = 1000;
flag = 0;
% for i=1:d
%     if d_r(i) > 0 && x_r(i) > 0 % TODO: modify to account for other constraints
%         ar_max = min(ar_max,  x_r(i) / d_r(i));
%     end
for t_i = 1:size(x_r,1)
    for t_y = 1:size(x_r,2)
        if d_r(t_i,t_y) < 0 && x_r(t_i,t_y) > act_thres
            ar_max = min(ar_max, x_r(t_i,t_y) / -d_r(t_i,t_y));
        end
    end
end
% end
% ar_max = norm(d_r(:));
x_r1 = x_r + ar_max * d_r;
lbd = 0.5;
if obj.func(x_r1) > obj.func(x_r)
    % call backtracking procedure
    ar = back_tracking(obj, ar_max, lbd, eps_h, x_r, d_r, is_grad);
    x_r1 = x_r + ar * d_r;
else
    flag = 1;
end
end
