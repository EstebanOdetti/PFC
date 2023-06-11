function [neighb] = fdm2d_neighbors(icone)
    % Armado de la matriz de vecindad a partir de la conectividad de los nodos
    % de la malla.
    maxn = max(max(icone));

    neighb = -1*ones(maxn,4);

    for i = 1 : size(icone,1)
        P1 = icone(i,1);
        P2 = icone(i,2);
        P3 = icone(i,3);
        P4 = icone(i,4);

        neighb(P1,2) = P2;
        neighb(P1,3) = P4;
        neighb(P2,3) = P3;
        neighb(P2,4) = P1;
        neighb(P3,1) = P2;
        neighb(P3,4) = P4;
        neighb(P4,1) = P1;
        neighb(P4,2) = P3;
    end
end