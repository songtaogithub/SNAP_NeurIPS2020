function [v] = spneon2(x, obj, act_set_x, grad)
% subspace perturbed gradient descent
beta = 1e-2;
F = 2e3;
T = 200;
y = 1e-4*randn(size(x));
y(act_set_x) = 0;
%proj = @(x)(reshape(sub_proj * x(:), size(x)));
%grad_ini = 
%v = proj(y);
%q_pi_x = zeros(size(x));
t = 0;
while t <= T
    y = y - beta * (obj.grad(x+y) - grad);
    y(act_set_x)= 0;
    t = t + 1;
end
v = y(:) / norm(y(:));
% grad(act_set_x) = 0 ;
% vc = v(~act_set_x)'*Hess*v(~act_set_x);
% if vc < 0
% if obj.func(x + y) - obj.func(x) - abs(grad(:)' *  y(:)) <= - F %&& norm(x-y) <= 0.9*(F)^(1/3)
%     flag = true;
%     vc = -0.1;
%     return
% else
%     v = zeros(size(v));
%     flag = false;
%     vc = 0;
% end
end

