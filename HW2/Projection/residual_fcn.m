function rho = residual_fcn(alpha,beta,delta,k_min,k_max,rho,grid_k,T_k,Z,PI,node_num,shock_num,M)

% Solves for the coefficients associated to the Chebyshev polynomials. 

    options = optimset('Display','Iter','TolFun',10^(-15),'TolX',10^(-15));
    rho = fsolve(@notime_iter,rho,options);

    function res = notime_iter(rho)

        residual_section = zeros(node_num,1);
        res = zeros(M,1);

        rho1 = rho(1:M,1);     % Coefficients for value function

        for z_index = 1:shock_num
    
            g_l   = zeros(node_num,1);
            g_k   = zeros(node_num,1);
            g_c   = zeros(node_num,1);
    
            rho1_section = rho1(((z_index-1)*node_num+1):z_index*node_num);
    
            for k_index = 1:node_num % Loop 1 over collocation point on k
        
                l = dot(rho1_section,T_k(k_index,:));          % labor at each collocation points
                k = grid_k(k_index);
                
                g_l(k_index) = l;

                y = exp(Z(z_index))*k^alpha*l^(1-alpha);
                c = (1-alpha)*y/l^2;

                kp = y+(1-delta)*k-c;
                if(kp < k_min)
                    kp = k_min+0.01;
                    disp('kp break lower bound')
                elseif((kp > y+(1-delta)*k-c) || (kp > k_max))
                    kp = min(y+(1-delta)*k-c,k_max) - 0.01;
                    disp('kp break upper bound')
                end
            
                g_k(k_index) = kp;

                c = y+(1-delta)*k-kp;
            
                if(c<0)
                    disp('warning: c < 0')
                    c = 10^(-10);
                end
            
                g_c(k_index) = c;

            end % Loop 1 over collocation point on k ends

            % scale k prime from [k_min,k_max] to [-1,1]
            g_k_scaled_down = (2*g_k-(k_min+k_max))/(k_max-k_min);
    
            % value of polynomials at each scaled k prime
            T_g_k = ones(node_num,node_num);
            T_g_k(:,2) = g_k_scaled_down;
            
            for i1=3:node_num
                T_g_k(:,i1) = 2*g_k_scaled_down.*T_g_k(:,i1-1)-T_g_k(:,i1-2);
            end
    
            % Calculate residual
            for k_index = 1:node_num % Loop 2 over collocation point on k

                temp = zeros(shock_num,1);
            
                for zp_index = 1:shock_num

                    rho1_section = rho1(((zp_index-1)*node_num+1):zp_index*node_num);
                    lp = dot(rho1_section,T_g_k(k_index,:));     

                    if(lp<0.01)
                        lp = 0.01;
                        disp('lp break lower bound')
                    elseif(lp>0.99)
                        lp = 0.99;
                        disp('lp break upper bound')
                    end

                    yp = exp(Z(zp_index))*g_k(k_index)^alpha*lp^(1-alpha);
                    cp = (1-alpha)*yp/lp^2;
                    kpp = yp+(1-delta)*g_k(k_index)-cp;

                    if(kpp<k_min)
                        kpp = k_min+0.01;
                        disp('kpp break lower bound')
                    elseif((kpp>yp+(1-delta)*g_k(k_index)-cp) || (kpp>k_max))
                        kpp = min(yp+(1-delta)*g_k(k_index)-cp,k_max) - 0.01;
                        disp('kpp break upper bound')
                    end

                    cp = yp+(1-delta)*g_k(k_index)-kpp;
                    if(cp<0)
                        disp('warning: cp < 0')
                        cp = 10^(-10);
                    end             

                    Ucp = cp^(-1);
                    Fkp = alpha*exp(Z(zp_index))*g_k(k_index)^(alpha-1)*lp^(1-alpha);
                    temp(zp_index) = Ucp*(Fkp+1-delta);
            
                end

                euler_rhs = beta*dot(PI(z_index,:),temp);

                c = g_c(k_index);
                
                euler_lhs = c^(-1);

                residual_section(k_index) = euler_rhs - euler_lhs;

            end % Loop 2 over k ends
    
            res(((z_index-1)*node_num+1):z_index*node_num) = residual_section;

         end

      end

  end