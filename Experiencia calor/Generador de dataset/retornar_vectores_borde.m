function  [DIR,hay_condicion_DIR,NEU,hay_condicion_NEU,ROB,hay_condicion_ROB,cant_casos_sur,cant_casos_este,cant_casos_norte,cant_casos_oeste]= retornar_vectores_borde(tipo_borde,DIR_values,NEU_values,ROB_values,nodos_Norte,nodos_Sur,nodos_Oeste,nodos_Este)
##tipo_borde=[Borde_sur,Borde_este,Borde_norte,Borde_oeste]
#Donde Borde_norte tomaria 1=Diritletch, 2=Neuman, 3=Robin

hay_condicion_ROB = ismember(3, tipo_borde);
hay_condicion_NEU = ismember(2, tipo_borde);
hay_condicion_DIR = ismember(1, tipo_borde);

DIR=[];
NEU=[];
ROB=[];

cant_casos_sur=0;
cant_casos_este=0;
cant_casos_norte=0;
cant_casos_oeste=1;

##borde_sur
tam_sur=length(nodos_Sur);
borde_sur_tipo_condicion=tipo_borde(1,1);
if (borde_sur_tipo_condicion==1)
    for i=1:length(DIR_values)
      valores_DIR=ones(tam_sur,1)*DIR_values(i);
        DIR=[DIR;nodos_Sur,valores_DIR];
        cant_casos_sur=cant_casos_sur+1;

    endfor
endif

if (borde_sur_tipo_condicion==2)
    for i=1:length(NEU_values)
      valores_NEU=ones(tam_sur,1)*NEU_values(i);
        NEU=[NEU;nodos_Sur,valores_NEU];
        cant_casos_sur=cant_casos_sur+1;
    endfor
endif

if (borde_sur_tipo_condicion==3)
    for i=1:length(NEU_values)
      valores_ROB=ones(tam_sur,1)*ROB_values(i);
        ROB=[ROB;nodos_Sur,valores_ROB];
        cant_casos_sur=cant_casos_sur+1;
    endfor
endif

##borde_este
tam_este=length(nodos_Este);
borde_este_tipo_condicion=tipo_borde(1,2);
if (borde_este_tipo_condicion==1)
    for i=1:length(DIR_values)
      valores_DIR=ones(tam_este,1)*DIR_values(i);
        DIR=[DIR;nodos_Este,valores_DIR];
        cant_casos_este=cant_casos_este+1;

    endfor
endif

if (borde_este_tipo_condicion==2)
    for i=1:length(NEU_values)
      valores_NEU=ones(tam_este,1)*NEU_values(i);
        NEU=[NEU;nodos_Este,valores_NEU];
        cant_casos_este=cant_casos_este+1;
    endfor
endif

if (borde_este_tipo_condicion==3)
    for i=1:length(NEU_values)
      valores_ROB=ones(tam_sur,1)*ROB_values(i);
        ROB=[ROB;nodos_Sur,valores_ROB];
        cant_casos_este=cant_casos_este+1;
    endfor
endif

##borde_norte
tam_norte=length(nodos_Norte);
borde_norte_tipo_condicion=tipo_borde(1,3);
if (borde_norte_tipo_condicion==1)
    for i=1:length(DIR_values)
      valores_DIR=ones(tam_norte,1)*DIR_values(i);
        DIR=[DIR;nodos_Norte,valores_DIR];
        cant_casos_norte=cant_casos_norte+1;

    endfor
endif

if (borde_norte_tipo_condicion==2)
    for i=1:length(NEU_values)
      valores_NEU=ones(tam_norte,1)*NEU_values(i);
        NEU=[NEU;nodos_Norte,valores_NEU];
        cant_casos_norte=cant_casos_norte+1;

    endfor
endif

if (borde_este_tipo_condicion==3)

    for i=1:length(NEU_values)
      valores_ROB=ones(tam_sur,1)*ROB_values(i);
        ROB=[ROB;nodos_Sur,valores_ROB];
        cant_casos_norte=cant_casos_norte+1;
    endfor
endif

##borde_oeste
tam_oeste=length(nodos_Norte);
borde_oeste_tipo_condicion=tipo_borde(1,4);
if (borde_oeste_tipo_condicion==1)
    for i=1:length(DIR_values)
      valores_DIR=ones(tam_oeste,1)*DIR_values(i);
        DIR=[DIR;nodos_Oeste,valores_DIR];
        cant_casos_oeste=cant_casos_oeste+1;

    endfor
endif

if (borde_oeste_tipo_condicion==2)
    for i=1:length(NEU_values)
      valores_NEU=ones(tam_oeste,1)*NEU_values(i);
        NEU=[NEU;nodos_Oeste,valores_NEU];
        cant_casos_oeste=cant_casos_oeste+1;

    endfor
endif

if (borde_este_tipo_condicion==3)

    for i=1:length(NEU_values)
      valores_ROB=ones(tam_sur,1)*ROB_values(i);
        ROB=[ROB;nodos_Sur,valores_ROB];
        cant_casos_oeste=cant_casos_oeste+1;
    endfor
endif

