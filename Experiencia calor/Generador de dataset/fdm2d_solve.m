function [PHI,Q] = fdm2d_solve(K,F,xnode,neighb,model)
    % Esquema temporal: [0] Explícito, [1] Implícito, [X] Estacionario
    if model.ts == 0 % Explícito
##        disp('Iniciando esquemas temporales...');
        % Paso temporal del método explícito Explícito (Forward Euler)
        [dt] = fdm2d_explicit_delta_t(xnode,model);
        [PHI,Q] = fdm2d_explicit(K,F,xnode,neighb,model,dt);
    elseif model.ts == 1 % Implícito
##        disp('Iniciando esquemas temporales...');
        % Paso temporal arbitrario
        dt = model.dt;
        [PHI,Q] = fdm2d_implicit(K,F,xnode,neighb,model,dt);
    elseif model.ts == 2 % Implícito crank nicolson
##         disp('Iniciando esquemas temporales...');
        % Paso temporal arbitrario
        dt = model.dt;
        [PHI,Q] = fdm2d_implicit_CN(K,F,xnode,neighb,model,dt);
    else % Estado estacionario
##        disp('Solving the equation system...');
        % Resolución del sistema lineal de ecuaciones
        PHI = K\F;

        % Cálculo del flujo de calor
        [Q] = fdm2d_flux(PHI,neighb,xnode,model.k);
    end
end

