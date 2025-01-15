function [kp,c,l,y,euler_error] = ... 
        eulerr_single(alpha,beta,delta,rho,...
        Z,PI,k_min,k_max,node_num,shock_num,M,z_index,k)

%  Computes Euler error for a given capital and productivity level.

    z = Z(z_index);

    k_scaled = 2*(k-k_min)/(k_max - k_min) -1;

    Tk = zeros(node_num,1);
    for i = 1:node_num % ith polynomial
        Tk(i) = cos(real(i-1)*acos(k_scaled));
    end

    rho1 = rho(1:M,1);    
    rho1_section = rho1(((z_index-1)*node_num+1):z_index*node_num);

    l = dot(rho1_section,Tk);       % Labor at each collocation points

    y = exp(z)*k^alpha*l^(1-alpha);
    c = (1-alpha)*y/l^2;          
    kp = y+(1-delta)*k-c;

    g_k_scaled = 2*(kp-k_min)/(k_max - k_min) -1;    
    T_g_k = zeros(node_num,1);

    for i = 1:node_num % ith polynomial
        T_g_k(i) = cos(real(i-1)*acos(g_k_scaled));
    end

    % calculate residual  
    fkp  = zeros(shock_num,1);
    temp = zeros(shock_num,1);

    for zp_index = 1:shock_num

        rho1_section = rho1(((z_index-1)*node_num+1):z_index*node_num);
        lp = dot(rho1_section,T_g_k);

        yp = exp(Z(zp_index))*kp^alpha*lp^(1-alpha);
        cp = (1-alpha)*yp/lp^2;

        Ucp = cp^(-1);
        fkp(zp_index) = alpha*exp(Z(zp_index))*kp^(alpha-1)*lp^(1-alpha);
        temp(zp_index) = Ucp*(fkp(zp_index)+1-delta);
    
    end    

    euler_rhs = beta*dot(PI(z_index,:),temp);

    euler_error = 1- euler_rhs*c;
    euler_error = log10(abs(euler_error));

    if( euler_error < -17 )
        euler_error = -17;
    end