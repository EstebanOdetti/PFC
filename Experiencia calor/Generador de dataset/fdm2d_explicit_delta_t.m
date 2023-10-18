function [dt] = fdm2d_explicit_delta_t(xnode,model)
    xcoord = sort(unique(xnode(:,1)));
    ycoord = sort(unique(xnode(:,2)));

    dx = abs(xcoord(1:end-1) - xcoord(2:end));
    dy = abs(ycoord(1:end-1) - ycoord(2:end));

    [delta,~] = min([dx;dy]);

    alpha = min(model.k)/(model.rho*model.cp);
    nd = 2;                     % porque estamos en 2D
    
    %% para el caso particular del TP delta tp muy pequeño
##    dt = 0.25*delta^2/(alpha*nd);
dt=0.00001;
    
end

