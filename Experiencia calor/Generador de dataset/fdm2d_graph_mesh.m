function [] = fdm2d_graph_mesh(PHI,Q,xnode,icone,mode,graph)
    if (mode == 0 || mode == 2)
        vm = 2;
    elseif (mode == 1 || mode == 3)
        vm = 3;
    end
    
    figure('Name', 'Resultados');
    
    err = 1;
    
    for i = 1 : size(PHI,2) - 1
        err = [err; norm(PHI(:,i+1)-PHI(:,i),2)/norm(PHI(:,i),2)];
    end
    
    X = xnode(:,1);
    Y = xnode(:,2);
    
    M = size(icone,1);
    triangles = zeros(2*M,3);
    
    for i = 1 : M
        triangles(i,:) = icone(i, 1:3);
        triangles(i+M,:) = [icone(i,1) icone(i,3) icone(i,4)];
    end

    %% Temperatura (escalar)
    if graph == 0
        zmin = min(min(PHI));
        zmax = max(max(PHI));
        
        for i = 1 : size(PHI,2)
            clf;
        	Z = PHI(:,i);
        
            hold on;
            patch('Vertices',[X Y Z],'Faces',triangles,'FaceVertexCData',Z);
            hold off;
            shading interp;

            if (mode < 2)
                hold on;
                patch('Vertices',[X Y Z],'Faces',icone,'FaceVertexCData',Z,...
                'FaceColor','none');
                hold off;
            end 
                
            if (i > 1)
                title(sprintf('nit: %d - error: %e',i-1,err(i)));
            end
            view(vm);
            zlim([zmin-0.001, zmax+0.001]);
            grid on;
            drawnow;
            pause(0.000001);
        end
        
        colorbar;
    end
    
    %% Flujo de calor (vectorial)
    if graph == 1
        maxq = max(max(abs(Q)));
        
        if (maxq > 0 && maxq < 5)
            scale = 2/maxq;
        elseif (maxq >= 5 && maxq < 10)
            scale = 1/maxq;
        elseif (maxq >= 10 && maxq < 100)
            scale = 5/maxq;
        elseif (maxq >= 100 && maxq < 1000)
            scale = 10/maxq;
        elseif (maxq >= 1000)
            scale = 100/maxq;
        else
            scale = 1;
        end
        
        xmax = max(xnode(:,1));
        xmin = min(xnode(:,1));
        ymax = max(xnode(:,2));
        ymin = min(xnode(:,2));
        zmax = max(max(PHI));
        zmin = min(min(PHI));
        
        for i = 1 : size(PHI,2)
            clf;
            Z = full(PHI(:,i));
            
            if (mode < 2)
                patch('Vertices',[X Y Z],'Faces',icone,'FaceVertexCData',Z,...
                    'FaceColor','none');
                shading interp;
            end
                
            hold on;
            Qx = Q(:,2*(i-1)+1);
            Qy = Q(:,2*(i-1)+2);
            Qz = zeros(size(Qx,1),1);
            quiver3(xnode(:,1), xnode(:,2), Z, Qx, Qy, Qz,...
                scale, 'color', 'black');
            hold off;
            if (i > 1)
                title(sprintf('nit: %d - error: %e',i-1,err(i)));
            end

            xlim([xmin-(xmax-xmin)*.1, xmax+(xmax-xmin)*.1]);
            ylim([ymin-(ymax-ymin)*.1, ymax+(ymax-ymin)*.1]);
            zlim([zmin-(zmax-zmin)*.1, zmax+(zmax-zmin)*.1]);
            grid on;
            view(vm);
            drawnow;
            pause(0.000001);
        end
        colorbar;
    end
    
    %% Flujo de calor en sentido eje-x (escalar)
    if graph == 2
        zmax = max(max(Q));
        zmin = min(min(Q));
        
        for i = 1 : size(PHI,2)
            clf;
            Z = Q(:,2*(i-1)+1);
            trisurf(triangles,X,Y,Z);
            shading interp;

            if (mode < 2)
                hold on;
                patch('Vertices',[X Y Z],'Faces',icone,'FaceVertexCData',Z,...
                'FaceColor','none');
                hold off;
            end
            
            if (i > 1)
                title(sprintf('nit: %d - error: %e',i-1,err(i)));
            end

            zlim([zmin zmax]);
            grid on;
            view(vm);
            drawnow;
            pause(0.000001);
        end
        colorbar;
    end
    
    %% Flujo de calor en sentido eje-y (escalar)
    if graph == 3
        zmax = max(max(Q));
        zmin = min(min(Q));
        
        for i = 1 : size(PHI,2)
            clf;
            Z = Q(:,2*(i-1)+2);
            trisurf(triangles,X,Y,Z);
            shading interp;

            if (mode < 2)
                hold on;
                patch('Vertices',[X Y Z],'Faces',icone,'FaceVertexCData',Z,...
                'FaceColor','none');
                hold off;
            end
            
            if (i > 1)
                title(sprintf('nit: %d - error: %e',i-1,err(i)));
            end

            zlim([zmin zmax]);
            grid on;
            view(vm);
            drawnow;
            pause(0.000001);
        end
        colorbar;
    end
    
    %% Magnitud del flujo de calor (escalar)
    if graph == 4
        Z = zeros(size(PHI));
        for i = 1 : size(PHI,2)
            Qx = Q(:,2*(i-1)+1);
            Qy = Q(:,2*(i-1)+2);
            
            for j = 1 : size(Qx,1)
                Z(j,i) = norm([Qx(j), Qy(j)],2);
            end
        end
        
        zmax = max(max(Z));
        zmin = min(min(Z));
        
        for i = 1 : size(PHI,2)
            clf;
            trisurf(triangles,X,Y,Z(:,i));
            shading interp;

            if (mode < 2)
                hold on;
                patch('Vertices',[X Y Z(:,i)],'Faces',icone,'FaceVertexCData',Z(:,i),...
                'FaceColor','none');
                hold off;
            end
            
            if (i > 1)
                title(sprintf('nit: %d - error: %e',i-1,err(i)));
            end

            zlim([zmin zmax]);
            grid on;
            view(vm);
            drawnow;
            pause(0.000001);
        end
        colorbar;
    end
end

