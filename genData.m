function [X, W, H] = genData(M, N, R, percent)
%UNTITLED4 Generate NMF data with sparse factors.
W = rand(M, R);
ind = randperm(M*R, floor(M*R*percent));
W(ind) = 0;
H = rand(N, R);
ind = randperm(N*R, floor(N*R*percent));
H(ind) = 0;
X = W*H';% + 1e-4 * randn(M, N);
end

