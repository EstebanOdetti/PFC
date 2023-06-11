function [PHI,Q] = fdm2d(xnode, icone, DIR, NEU, ROB, model)
    % - xnode : matriz de pares (x,y). Cada fila contiene las coordenadas
    %   x e y de un nodo de la malla. El número de fila (índice) corresponde
    %   al número de nodo.
    % - icone : matriz de conectividad. Cada fila contiene los índices de
    %   los nodos que integran un elemento, típicamente los 4 nodos que forman
    %   un cuadrángulo.

    % Inicialización de variables principales del sistema
    [K,F] = fdm2d_initialize(model.nnodes);

    % Armado de la matriz de vecindad
    [neighb] = fdm2d_neighbors(icone);
    
    % Ensamble de coeficientes del sistema
    [K,F] = fdm2d_gen_system(K,F,xnode,neighb,model.k,model.c,model.G);

    % Ensamble de nodos frontera Neumann
    [F] = fdm2d_neumann(F,xnode,neighb,NEU);
    
    % Ensamble de nodos frontera Robin
    [K,F] = fdm2d_robin(K,F,xnode,neighb,ROB);
    
    % Ensamble de nodos frontera Dirichlet
    [K,F] = fdm2d_dirichlet(K,F,DIR);
    
    % Resolución del sistema lineal de ecuaciones
    [PHI,Q] = fdm2d_solve(K,F,xnode,neighb,model);
    
    
end

