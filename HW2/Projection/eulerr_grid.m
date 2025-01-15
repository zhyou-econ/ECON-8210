function [g_k,g_c,g_l,euler_error,max_error]= ...
         eulerr_grid(alpha,beta,delta,rho,Z, ...
         PI,k_min,k_max,grid_k_complete,shock_num,node_num,grid_num,M)

% This function computes the Euler Errors on the capital and exogenous shock grid

    grid_k_complete_scaled = (2*grid_k_complete-(k_min+k_max))/(k_max-k_min);

    T_k_complete = ones(grid_num,node_num);
    T_k_complete(:,2) = grid_k_complete_scaled;
    
    for i1 = 3:node_num
        T_k_complete(:,i1) = 2*grid_k_complete_scaled.*T_k_complete(:,i1-1)-T_k_complete(:,i1-2);
    end     

    rho1 = rho(1:M,1);    

    euler_error = zeros(grid_num,shock_num);
    g_l = zeros(grid_num,shock_num);
    g_c = zeros(grid_num,shock_num);
    g_k = zeros(grid_num,shock_num);
    
    for z_index = 1:shock_num
        
        for k_index = 1:grid_num % Loop 1 over collocation point on k

            rho1_section = rho1(((z_index-1)*node_num+1):z_index*node_num);
            g_l(k_index,z_index) = dot(rho1_section,T_k_complete(k_index,:));       % Labor at each collocation points

            y = exp(Z(z_index))*grid_k_complete(k_index)^alpha*g_l(k_index,z_index)^(1-alpha);
            g_c(k_index,z_index) = (1-alpha)*y/g_l(k_index,z_index)^2;
            g_k(k_index,z_index) = y+(1-delta)*grid_k_complete(k_index)-g_c(k_index,z_index);            

        end % Loop 1 over collocation point on k ends

        % Scale k prime from [k_min,k_max] to [-1,1]
        g_k_scaled_down = (2*g_k(:,z_index)-(k_min+k_max))/(k_max-k_min);
        
        % value of polynomials at each scaled k prime
        T_g_k = ones(grid_num,node_num);
        T_g_k(:,2) = g_k_scaled_down;
        
        for i1 = 3:node_num
            T_g_k(:,i1) = 2*g_k_scaled_down.*T_g_k(:,i1-1)-T_g_k(:,i1-2);
        end     

        % Calculate residual        
        for k_index = 1:grid_num % Loop 2 over collocation point on k              
            
            temp = zeros(shock_num,1);
        
            for zp_index = 1:shock_num
            
                rho1_section = rho1(((zp_index-1)*node_num+1):zp_index*node_num);
                lp = dot(rho1_section,T_g_k(k_index,:));

                yp = exp(Z(zp_index))*g_k(k_index,z_index)^alpha*lp^(1-alpha);
                cp = (1-alpha)*yp/lp^2;

                Ucp = cp^(-1);
                Fkp = alpha*exp(Z(zp_index))*g_k(k_index,z_index)^(alpha-1)*lp^(1-alpha);
                temp(zp_index) = Ucp*(Fkp+1-delta);
            
            end
            
            euler_rhs = beta*dot(PI(z_index,:),temp);

            euler_error(k_index,z_index) = 1- euler_rhs*g_c(k_index,z_index);

        end % Loop 2 over k ends

    end % Loop over z ends

    euler_error = log10(abs(euler_error));
    max_error = max(euler_error,[],1);