function ss = bt(x, obj, dr)
ss = 1; % default
alpha = 0.5;
beta = 0.5;
grad_x = obj.grad(x);
fx = obj.func(x);
it = 0;
while obj.func(x + ss*dr) > fx + alpha * ss * dr(:)'*grad_x(:)
    ss = ss * beta;
    it = it + 1;
end
% z = 1;
end
