function alpha = back_tracking(obj, alpha_0, lbd, eps_h, x_r, d_r, is_grad)
% grad tell if using gradient direction or negative curvature direction
eps_h_p = eps_h; % TODO
if is_grad
    ndr = norm(d_r(:))^2;
    rho_alpha = @(alpha)(alpha * ndr);
else
    rho_alpha = @(alpha)(alpha^2 * eps_h_p / 4);
end
alpha = alpha_0;
f_r = obj.func(x_r);
while obj.func(x_r + alpha * d_r) > f_r - lbd * rho_alpha(alpha)
    alpha = alpha * 1/2;
end
end
