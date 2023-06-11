function [K,F] = fdm2d_dirichlet(K,F,DIR)
% Descripción: módulo para calcular y ensamblar las contribuciones de nodos 
% pertenecientes a fronteras de tipo Dirichlet.

% Entrada:
% * K: matriz del sistema (difusión + reacción)
% * F: vector de flujo térmico.
% * DIR: matriz con la información sobre la frontera de tipo Dirchlet.
%   - Columna 1: número de nodo.
%   - Columna 2: valor en ese nodo (escalar)

% Salida:
% * K: matriz del sistema (difusión + reacción) luego de realizar las simplificaciones
%   que surgen de aplicar la condición de borde Dirichlet.
% * F: vector de flujo térmico luego de realizar las simplificaciones que surgen de
%   aplicar la condición de borde Dirichlet.
% ----------------------------------------------------------------------
    M = size(DIR);
    for n = 1 : M
        p=DIR(n);
        phi=DIR(n,2);
        K(p,:)=0;
        K(p,p)=1;
        F(p)=phi;      
    end
end


