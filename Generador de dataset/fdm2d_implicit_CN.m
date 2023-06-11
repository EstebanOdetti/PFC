function [PHI_vec, Q_vec] = fdm2d_implicit_CN(K,F,xnode,neighb,model,dt)
% Descripción: módulo para resolver el sistema lineal de ecuaciones utilizando esquema
% temporal implícito.

% Entrada:
% * K: matriz del sistema (difusión + reacción)
% * F: vector de flujo térmico.
% * xnode: matriz de nodos con pares (x,y) representando las coordenadas de cada nodo 
% de la malla.
% * neighb: matriz de vecindad.
% * model: struct con todos los datos del modelo (constantes, esquema numérico, etc.)
% * dt: paso temporal crítico para método explícito.

% Salida:
% * PHI: matriz solución. Cada elemento del vector representa un valor escalar 
%   asociado a cada nodo de la malla, y su posición dentro del vector depende de
%   cómo se especificó cada nodo en xnode. Cada columna representa una iteración
%   del esquema temporal (en total nit columnas).
% * Q: matriz de flujo de calor. Para cada nodo se halla un vector bidimensional
%   de flujo de calor, representado por un par (Qx,Qy). Cada par de columnas 
%   representa una iteración del esquema temporal (en total 2×nit columnas).
% ----------------------------------------------------------------------

    A = (model.rho*model.cp)/dt;
    
    I = A*eye(model.nnodes,model.nnodes);
    inversa_K = sparse(model.nnodes,model.nnodes);
    PHI = model.PHI_n;
    PHI_n = model.PHI_n;
    PHI_vec = PHI;
    Q_vec = zeros(model.nnodes,2);
    length(PHI_n)

    inversa_K=inv(I+(0.5*K));
    for n = 1 : model.maxit
    PHI = inversa_K*(((I-(0.5*K))*PHI_n)+F);
    err = norm(PHI-PHI_n,2)/norm(PHI,2);
    PHI_n = PHI;
    PHI_vec = [PHI_vec PHI];
    [Q] = fdm2d_flux(PHI,neighb,xnode,model.k);
    Q_vec = [Q_vec, Q];
        
        if err < model.tol
            disp('Método terminado por tolerancia de error.');
            return;
        end
    end

    disp('Método terminado por límite de iteraciones.');
 
end