function M = Knn(n)
% form the commutation matrix of size n * n

ind1 = 1:n^2;
ind2 = reshape(ind1, n, n)';
ind2 = ind2(:);

M = sparse(ind1, ind2, 1, n^2, n^2);
end

