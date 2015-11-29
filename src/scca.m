function [ U,V ] = scca( K,U_init,V_init,lambda_u,lambda_v )
% sparse cca
% input: K - pxq covariance matrix
%        U_init - px1 initial U
%        V_init - qx1 initial V
%        lambda_u - sparsity para for U
%        lambda_v - sparsity para for V
% output: U - px1
%         V - qx1 

eps = 0.001;
max_iter = 50;
diff_u=eps*10;
diff_v=eps*10;

for i=1:max_iter
    if diff_u<eps && diff_v<eps, break, end
    
    % Update left singular vector
    U=K*V_init;
    % normalize U
    V_norm=norm(U);
    if V_norm == 0, V_norm=1; end
    U=U/V_norm;
    % soft shresholding
    U_sign=sign(U);
    U_sign(U_sign==0)=-1;
    U=abs(U)-0.5*lambda_u;
    U=(U+abs(U))/2;
    U=U.*U_sign;
    % normalize U
    V_norm=norm(U);
    if V_norm == 0, V_norm=1; end
    U=U/V_norm;
    
    % Update right singular vector
    V=K'*U;
    % normalize V
    V_norm=norm(V);
    if V_norm == 0, V_norm=1; end
    V=V/V_norm;
    % soft shresholding
    V_sign=sign(V);
    V_sign(V_sign==0)=-1;
    V=abs(V)-0.5*lambda_v;
    V=(U+abs(V))/2;
    V=V.*V_sign;
    % normalize V
    V_norm=norm(V);
    if V_norm == 0, V_norm=1; end
    V=V/V_norm;
    
    % Convergence measures 
    diff_u=max(abs(u_initial - U));
    diff_v=max(abs(v_initial - V));
    
    v_initial = V;
	u_initial = U;
end


end

