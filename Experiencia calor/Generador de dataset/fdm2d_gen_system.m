function [K,F] = fdm2d_gen_system(K,F,xnode,neighb,k,c,G)
% Descripción: módulo para ensamblar los términos difusivo, reactivo y fuente de
% todos los nodos de la malla, generando el stencil adecuado dependiendo de si es
% un nodo interior o de frontera.

% Entrada:
% * K: matriz del sistema (difusión + reacción)
% * F: vector de flujo térmico.
% * xnode: matriz de pares (x,y) representando cada nodo de la malla.
% * neighb: matriz de vecindad.
% * k: conductividad térmica del material. Es un vector que permite representar k(x,y).
% * c: constante de reacción del material. Es un vector que permite representar c(x,y).
% * G: fuente de calor. Es un vector que permite representar G(x,y).

% Salida:
% * K: matriz del sistema (difusión + reacción) con modificaciones luego del ensamble.
% * F: vector de flujo térmico con modificaciones luego del ensamble.
% ----------------------------------------------------------------------


    for P = 1 : size(xnode,1)
        % 1) de la estructura neighb obtener los cuatro vecinos de cada
        % nodo P.          
        S = neighb(P, 1);        
        E = neighb(P, 2);       
        N = neighb(P, 3);        
        W = neighb(P, 4);     
        

        % 2) obtener la distancia (a partir de la esctructura xnode) de 
        % cada nodo P hacia sus cuatro vecinos. 
        % Si es un nodo de borde (es decir que neighb es -1 en alguna de
        % sus columnas) no definir esa distncia. Luego en el stencil 
        % considerar el reemplazo del nodo ficticio tomando el stencil 
        % clasico para la derivada segunda en esa direccion.
        ds = 0; de = 0; dn = 0; dw = 0;

        if(S != -1)
            ds = abs(xnode(S,2) - xnode(P,2));
        end
        if(E != -1)
            de = abs(xnode(E,1) - xnode(P,1));
        end
        if(N != -1)
            dn = abs(xnode(N,2) - xnode(P,2));
        end
        if(W != -1)
            dw = abs(xnode(W,1) - xnode(P,1));
        end        
        
        % 3) Considerando d2T/dx2 = ax*T(i+1,j)+bx*T(i,j)+cx*T(i-1,j) y d2T/dy2 = ay*T(i,j+1)+by*T(i,j)i+cy*T(i,j-1)
        % calcular los coeficientes asociados a cada nodo vecino en base a las distancias calculadas en el paso anterior y asi definir el stencil.
        if (W == -1)
            ax = 2/(de*de);
            bx = -2/(de*de);
            cx = 0;
        elseif (E == -1)
            ax = 0;
            bx = -2/(dw*dw);
            cx = 2/(dw*dw);
        else
            ax = 2/(de*(de+dw));
            bx = -2/(de*dw);
            cx = 2/(dw*(de+dw));
        end
        if (S == -1)
            ay = 2/(dn*dn);
            by = -2/(dn*dn);
            cy = 0;
        elseif (N == -1)
            ay = 0;
            by = -2/(ds*ds);
            cy = 2/(ds*ds);
        else
            ay = 2/(dn*(dn+ds));
            by = -2/(ds*dn);
            cy = 2/(ds*(ds+dn));
        end
        K(P,P) = c(P)-(k(P)*bx)-(k(P)*by);
        
        

        if(E != -1)
        K(P,E)=-k(P)*ax;
        end
        if(W != -1)
        K(P,W)=-k(P)*cx;
        end    
        if(S != -1)
        K(P,S)=-k(P)*cy;
        end
        if(N != -1)
        K(P,N)=-k(P)*ay;
        end

             
        F(P)=G(P);

    end

    
   
end