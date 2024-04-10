function [dataset_salida] = reordenar_a_matriz(dataset,cant_filas)
  length(dataset(:,1));
  cant_columnas=floor(length(dataset(:,1))/7);
  dataset_salida=zeros(cant_columnas,cant_columnas,20);
    for i=1:cant_columnas
      dataset_salida(:,i,:)=dataset(((i-1)*cant_filas)+1:(i)*cant_filas,:);
    endfor
endfunction
