function [dataset_salida] = reordenar_a_matriz(dataset,cant_filas)

  ##falta terminar me estoy enredando con la fomra de la matriz
  ##al ser una malla cuadrada tengo 7 columnas por 7 filas
  ##por 20 datos de entradas en cada elemento de la matriz
  length(dataset(:,1));
cant_columnas=floor(length(dataset(:,1))/7);
dataset_salida=zeros(cant_columnas,cant_columnas,20);
    for i=1:cant_columnas
##      inicio_fila=((i-1)*cant_filas)+1
##      fin_fila=(i)*cant_filas
##      display("------")
##      full(dataset(((i-1)*cant_filas)+1:(i)*cant_filas,:))'
      dataset_salida(:,i,:)=dataset(((i-1)*cant_filas)+1:(i)*cant_filas,:);
    endfor
##una_salida=dataset_salida(1,1,:)
endfunction
