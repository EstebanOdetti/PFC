function [K,F] = fdm2d_initialize(nnodes)
    % Inicialización de la matriz del sistema, K, y el vector de flujo térmico, F.
    K = sparse(nnodes,nnodes);
    F = sparse(nnodes,1);
end

