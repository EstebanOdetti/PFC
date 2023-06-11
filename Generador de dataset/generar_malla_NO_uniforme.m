function [xnode,icone,DIR,NEU,ROB]=generar_malla_NO_uniforme(ejex,ejey,CBN,CBS,CBE,CBO)
  %%ejex e y tiene el pricipio y final de los seegmentos ademas del delta en cada uno
  cant_pun_y=0;
  cant_pun_x=0;
  posx=ejex(1,1);
  posy=ejey(1,1);
  resetx=0;
  can_secc_x=rows(ejex);
  can_secc_y=rows(ejey);
  aux=0;
  deltax=generar_desplazaminetos_indice(ejex);
  deltay=generar_desplazaminetos_indice(ejey);
  if(can_secc_x==1)
  aux=1
  endif

  for i=1:can_secc_x
  cant_pun_x=((ejex(i,2)-ejex(i,1))/ejex(i,3))+cant_pun_x;
  endfor
  for i=1:can_secc_y
  cant_pun_y=((ejey(i,2)-ejey(i,1))/ejey(i,3))+cant_pun_y;
  endfor
if((floor(cant_pun_x)-cant_pun_x)!=0)
fprintf("DIVISION NO EXACTA EN X")
cant_pun_x

endif
if((floor(cant_pun_y)-cant_pun_y)!=0)
fprintf("DIVISION NO EXACTA EN Y")
cant_pun_y

endif
  contador=1;
  contador_icone=1;
  contador_DIR_N=1;
  contador_DIR_S=1;
  contador_DIR_O=1;
  contador_DIR_E=1;
  
  contador_NEU_N=1;
  contador_NEU_S=1;
  contador_NEU_O=1;
  contador_NEU_E=1;
  
  contador_ROB_N=1;
  contador_ROB_S=1;
  contador_ROB_O=1;
  contador_ROB_E=1;

  xnode=zeros(1,2);
  icone=zeros(1,4);

  DIR=[];
  DIR_N=[];
  DIR_S=[];
  DIR_O=[];
  DIR_E=[];
  
  NEU=[];
  NEU_N=[];
  NEU_S=[];
  NEU_O=[];
  NEU_E=[];
  
  ROB=[];
  ROB_N=[];
  ROB_S=[];
  ROB_O=[];
  ROB_E=[];
  
  for i=1:cant_pun_y+1
    
    for j=1:cant_pun_x+1
            
    xnode(contador,:)=[posx posy]; 
    if(j!=(cant_pun_x+1)&&i!=(cant_pun_y+1))
      icone(contador_icone,:)=[contador (contador+1) (contador+cant_pun_x+2) (contador+cant_pun_x+1)];
      contador_icone=contador_icone+1;
    endif
  %%frotera sur j=1,valDIRen el 1 es la sur
       if(i==1)
      for k=1:rows(CBS)
       if(CBS(k,2)<=posx && CBS(k,3)>=posx)
         if(CBS(k,1)==1)%cond diritlech
            DIR_S(contador_DIR_S,:)=[contador CBS(k,4)];
            contador_DIR_S=contador_DIR_S+1;
            break
         endif
         if(CBS(k,1)==2)%cond neuman
           NEU_S(contador_NEU_S,:)=[contador CBS(k,4) 1];
           contador_NEU_S=contador_NEU_S+1;            
            break
         endif
         if(CBS(k,1)==3)%cond robin
           ROB_S(contador_ROB_S,:)=[contador CBS(k,4) CBS(k,5) 1];
           contador_ROB_S=contador_ROB_S+1;
         endif
        endif 
       endfor
  endif
   %%frotera norte j=N+1,valDIRen el 3 es la norte
    if(i==(cant_pun_y+1))
    for k=1:rows(CBN)
       if(CBN(k,2)<=posx && CBN(k,3)>=posx)
          if(CBN(k,1)==1)%cond diritlech
            DIR_N(contador_DIR_N,:)=[contador CBN(k,4)];
            contador_DIR_N=contador_DIR_N+1;
             break
          endif
          if(CBN(k,1)==2)%cond newmann
            NEU_N(contador_NEU_N,:)=[contador CBN(k,4) 3];
            contador_NEU_N=contador_NEU_N+1;
            break
          endif
          if(CBN(k,1)==3)%cond robin
            ROB_N(contador_ROB_N,:)=[contador CBN(k,4) CBN(k,5) 3];
            contador_ROB_N=contador_ROB_N+1;
            break
          endif
       endif 
     endfor
  endif
     %%frotera oeste i=1,valDIRen el 4 es la oeste
   if(j==1)
    for k=1:rows(CBO)
       if(CBO(k,2)<=posy&& CBO(k,3)>=posy)
          if(CBO(k,1)==1)%cond diritlech
             DIR_O(contador_DIR_O,:)=[contador CBO(k,4)];
             contador_DIR_O=contador_DIR_O+1;
             break
           endif
           
           if(CBO(k,1)==2)%cond newmann
             NEU_O(contador_NEU_O,:)=[contador CBO(k,4) 4];
             contador_NEU_O=contador_NEU_O+1;
             break
           endif
 
           if(CBO(k,1)==3)%cond robin
                  ROB_O(contador_ROB_O,:)=[contador CBO(k,4) CBO(k,5) 4];
                  contador_ROB_O=contador_ROB_O+1;
            endif
         endif
     endfor
  endif
     %%frotera este i=N+1,valDIRen el 4 es la este
   if(j==(cant_pun_x+1))
    for k=1:rows(CBE)
       if(CBE(k,2)<=posy && CBE(k,3)>=posy)
        if(CBE(k,1)==1)%cond diritlech
           DIR_E(contador_DIR_E,:)=[contador CBE(k,4)];
           contador_DIR_E=contador_DIR_E+1;
           break
        endif
        if(CBE(k,1)==2)%cond newmann
           NEU_E(contador_NEU_E,:)=[contador CBE(k,4) 2];
           contador_NEU_E=contador_NEU_E+1;
           break
        endif
        if(CBE(k,1)==3)%cond robin
           ROB_E(contador_ROB_E,:)=[contador CBE(k,4) CBE(k,5) 2];
           contador_ROB_E=contador_ROB_E+1;
        endif
      endif
     endfor
  endif
    contador=contador+1;

             if(posx==ejex(end,2)||j==cant_pun_x+1)
                      posx=ejex(1,1);
                      
              else
                 posx=deltax(j)+posx;      
              endif
           

  endfor

          posy=posy+deltay(i);
  endfor
  

  DIR=[DIR_N;DIR_S;DIR_O;DIR_E];
  NEU=[NEU_E;NEU_O;NEU_N;NEU_S]; 
 
  ROB=[ROB_E;ROB_O;ROB_N;ROB_S];
  
endfunction