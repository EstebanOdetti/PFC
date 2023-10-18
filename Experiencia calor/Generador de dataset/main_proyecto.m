function [PHI,Q] = main_proyecto(xnode, icone, DIR, NEU, ROB, G, k, c)
model.G = G;
model.k= k;
model.c = c;

##disp('---------------------------------------------------------------');
##disp('Inicializando modelo de datos...');

model.nnodes = size(xnode,1);


% Esquema Temporal: [0] Explícito, [1] Implícito, [X] Estacionario
model.ts = 3;

% Parámetros para esquemas temporales
model.rho = 1.0000000000000000;
model.cp = 1.0000000000000000;
model.maxit =            1;
model.tol = 1.000000e-05;
model.dt = 0.5;
% Condición inicial
model.PHI_n = mean(DIR(:,2))*ones(model.nnodes,1);

##disp('Iniciando el método numérico...');

% Llamada principal al Método de Diferencias Finitas
[PHI,Q] = fdm2d(xnode, icone, DIR, NEU, ROB, model);
##
##disp('Finalizada la ejecución del método numérico.');
##
##disp('---------------------------------------------------------------');
##disp('Iniciando el post-procesamiento...');

% mode ---> modo de visualización:
%           [0] 2D - Con malla
%           [1] 3D - Con malla
%           [2] 2D - Sin malla
%           [3] 3D - Sin malla
% graph --> tipo de gráfica:
%           [0] Temperatura (escalar)
%           [1] Flujo de Calor (vectorial)
%           [2] Flujo de Calor eje-x (escalar)
%           [3] Flujo de Calor eje-y (escalar)
%           [4] Magnitud de Flujo de Calor (escalar)
##mode = 0;
##graph = 0;
##fdm2d_graph_mesh(full(PHI),Q,xnode,icone,mode,graph);

##disp('Finalizado el post-procesamiento.');
end
