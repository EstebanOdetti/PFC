function [F] = fdm2d_neumann(F,xnode,neighb,NEU)
% Descripción: módulo para calcular y ensamblar las contribuciones de nodos
% pertenecientes a fronteras de tipo Neumann.

% Entrada:
% * F: vector de flujo térmico.
% * xnode: matriz de nodos con pares (x,y) representando las coordenadas de cada
%   nodo de la malla.
% * neighb: matriz de vecindad.
% * NEU: matriz con la información sobre la frontera de tipo Neumann.
%   - Columna 1: índice del nodo donde se aplica la condición de borde.
%   - Columna 2: valor de flujo térmico (q) asociado al lado del elemento.
%   - Columna 3: dirección y sentido del flujo:
%     (1) Flujo en dirección eje-y, sentido negativo (S – South - Sur)
%     (2) Flujo en dirección eje-x, sentido positivo (E – East - Este)
%     (3) Flujo en dirección eje-y, sentido positivo (N – North - Norte)
%     (4) Flujo en dirección eje-x, sentido negativo (W – West – Oeste)

% Salida:
% * F: vector de flujo térmico con modificaciones luego de aplicar la condición de borde.
% ----------------------------------------------------------------------
    M = size(NEU, 1);
    for n = 1 : M
        P = NEU(n, 1);
        S = neighb(P, 1);
        E = neighb(P, 2);
        N = neighb(P, 3);
        W = neighb(P, 4);

        q = NEU(n,2);

        if (NEU(n,3) == 1)
            dy = abs(xnode(N,2) - xnode(P,2));
            F(P) = F(P) - 2*q/dy;
        end

        if (NEU(n,3) == 2)
            dx = abs(xnode(W,1) - xnode(P,1));
            F(P) = F(P) - 2*q/dx;
        end

        if (NEU(n,3) == 3)
            dy = abs(xnode(S,2) - xnode(P,2));
            F(P) = F(P) - 2*q/dy;
        end

        if (NEU(n,3) == 4)
            dx = abs(xnode(E,1) - xnode(P,1));
            F(P) = F(P) - 2*q/dx;
        end
    end
end
