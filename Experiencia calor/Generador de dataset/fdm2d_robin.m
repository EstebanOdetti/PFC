function [K,F] = fdm2d_robin(K,F,xnode,neighb,ROB)
% Descripción: módulo para calcular y ensamblar las contribuciones de nodos 
% pertenecientes a fronteras de tipo Robin.

% Entrada:
% * K: matriz del sistema (difusión + reacción)
% * F: vector de flujo térmico.
% * xnode: matriz de nodos con pares (x,y) representando las coordenadas de 
%   cada nodo de la malla.
% * neighb: matriz de vecindad.
% * ROB: matriz con la información sobre la frontera de tipo Robin.
%   - Columna 1: índice del nodo donde se aplica la condición de borde.
%   - Columna 2: valor de coeficiente de calor (h)
%   - Columna 3: valor de temperatura de referencia (phi_inf).
%   - Columna 4: dirección y sentido del flujo:
%     (1) Flujo en dirección eje-y, sentido negativo (S – South – Sur)
%     (2) Flujo en dirección eje-x, sentido positivo (E – East – Este)
%     (3) Flujo en dirección eje-y, sentido positivo (N – North – Norte)
%     (4) Flujo en dirección eje-x, sentido negativo (W – West – Oeste)

% Salida:
% * K: matriz del sistema (difusión + reacción) con modificaciones luego de 
%   aplicar la condición de borde.
% * F: vector de flujo térmico con modificaciones luego de aplicar la condición 
%   de borde.
% ----------------------------------------------------------------------

    M = size(ROB, 1);
    for n = 1 : M
        P = ROB(n, 1);           
        S = neighb(P, 1);        
        E = neighb(P, 2);       
        N = neighb(P, 3);        
        W = neighb(P, 4);        

        h = ROB(n,2);
        phi_inf = ROB(n,3);
        valor=2*h*phi_inf;
                
        if (ROB(n,4) == 1)
            dy = abs(xnode(N,2) - xnode(P,2));
            F(P) = F(P) + valor/dy;
            K(P,P)=K(P,P)+(2*h/dy);
        end
        
        if (ROB(n,4) == 2) 
            dx = abs(xnode(W,1) - xnode(P,1));
            F(P) = F(P) + valor/dx;
            K(P,P)=K(P,P)+(2*h/dx);
        end
        
        if (ROB(n,4) == 3) 
            dy = abs(xnode(S,2) - xnode(P,2));
            F(P) = F(P) + valor/dy;
            K(P,P)=K(P,P)+(2*h/dy);
        end
        
        if (ROB(n,4) == 4)
            dx = abs(xnode(E,1) - xnode(P,1));
            F(P) = F(P) + valor/dx;
            K(P,P)=K(P,P)+(2*h/dx);
        end
    end
end