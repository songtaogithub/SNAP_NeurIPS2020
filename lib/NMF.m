classdef NMF
    properties
        X
        I % rows
        J % cols
        R = 5
    end
    methods
        function obj = NMF(X, R)
            % constructor
            obj.X = X;
            obj.I = size(X, 1);
            obj.J = size(X, 2);
            obj.R = R;
        end
        
        function cost = func(obj, WH)
        % compute NMF LS cost, assumes W, H is feasible
            W = WH(1:obj.I, :);
            H = WH(obj.I+1:end, :);
            cost = 0.5 * norm(obj.X - W*H', 'fro')^2;
        end
        
        function gradient = grad(obj, WH)
            %  compute NMF LS grad, assumes W, H is feasible
            W = WH(1:obj.I, :);
            H = WH(obj.I+1:end, :);
            E = (W*H' - obj.X);
            gradient = [E  * H; E' * W];
        end 
        
        function Hess = hessian(obj, WH)
            W = WH(1:obj.I, :);
            H = WH(obj.I+1:end, :);
            
            % TODO: naive implementation
            WpH = W*H';
            H11 = kron(H'*H, eye(obj.I));
            H12 = -kron(eye(obj.R), obj.X) + kron(eye(obj.R),WpH) + ...
                   kron(eye(obj.R), W) * Knn(obj.R) *  kron(eye(obj.R), H');
            H21 = -kron(eye(obj.R), obj.X') + kron(eye(obj.R), WpH') + ...
                   kron(eye(obj.R), H) * Knn(obj.R) *  kron(eye(obj.R), W');
            H22 = kron(W'*W, eye(obj.J));
            
            Hess = [H11, H12; H21, H22];
            
            
%             Hess = [kron(H'*H, eye(obj.I)), -kron(eye(obj.R), obj.X); -kron(eye(obj.R), obj.X'),...
%                     kron(W'*W, eye(obj.J))];
            
        end
    end
end

