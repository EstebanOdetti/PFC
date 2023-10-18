function [in_despl]=generar_desplazaminetos_indice(eje)
  cant_elem=rows(eje);
  fin=((eje(1,2)-eje(1,1))/eje(1,3));
  in_despl(1:fin)=eje(1,3);
  for i=2:cant_elem
      fin=fin+1;
      cant_local=((eje(i,2)-eje(i,1))/eje(i,3));
      in_despl(fin:(fin+cant_local-1))=eje(i,3);
      fin=cant_local+fin-1;
  endfor
  in_despl(fin+1)=0.5;in_despl(fin+1)=0.5;
endfunction
