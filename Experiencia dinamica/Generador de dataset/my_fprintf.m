function my_fprintf(filename,format,data)
% rutina propia que se encaraga de escribir
% usando rutinas de matlab
%     fopen - fprintf - fclose
% un conjunto de datos "data"
% con un formato "format"
% y en un fichero de nombre "filename"
%
% my_fprintf(filename,format,data)
%
% format es del tipo C
% por si no lo recuerda algo asi como
%     '%6.2f  %12.8f\n'
%     '%5i%5i%5i%5i \n' si quiere algo de 4 campos de i5 
%                          en cada registro
% En en caso del programa tango usar:
%
%    para COOR '%5i     %17.9e%17.9e%17.9e \n',
%    para ELEM '%5i%5i%5i%5i%5i%5i%5i%5i%5i%5i \n',
%
%           VER tambien FPRINTF FOPEN FCLOSE

data=data';
%eval(['!rm ' filename ])
fid = fopen(filename,'w');
fprintf(fid,format,data);
fclose(fid);