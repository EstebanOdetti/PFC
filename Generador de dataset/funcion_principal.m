clc;
pkg load io

warning('off','all');
display("inicio")

##Declaramos las variables que utilizaremos
xnode = retornar_xnode();
icone =retornar_icone;
model.k = retornar_K();
model.c = retornar_C();
model.G = retornar_G();
dataset = [];

salida_vector=[];
borde = zeros([length(xnode),2]);
columna_DIR=zeros([length(xnode),2]);
columna_NEU=zeros([length(xnode),2]);
columna_ROB=zeros([length(xnode),2]);
mascara_tipos_bordes_dataset=zeros([length(xnode),4]);
nodos_Norte=retornar_nodos_norte();
nodos_Sur=retornar_nodos_sur();
nodos_Oeste=retornar_nodos_oeste();
nodos_Este=retornar_nodos_este();



##[DIR_2,hay_condicion_DIR,NEU_2,hay_condicion_NEU,ROB_2,hay_condicion_ROB,cant_casos_sur,cant_casos_este,cant_casos_norte,cant_casos_oeste]= retornar_vectores_borde(tipo_borde,DIR_values,NEU_values,ROB_values,nodos_Norte,nodos_Sur,nodos_Oeste,nodos_Este);


## Anotacion para las condiciones
##        S = neighb(P, 1);
##        E = neighb(P, 2);
##        N = neighb(P, 3);
##        W = neighb(P, 4);

##DEfinimos los valores que van a recorrer

##PAra ir variando el tipo de condicion de borde de los 4 borde
##propongo usar un vector Asi saber que condicion es cada una

#Donde Borde_norte tomaria 1=Diritletch, 2=Neuman, 3=Robin



PHI = [];
Q = [];
G_values = [300:100:600];
##CONDUCTIVIDAD TERMICA ALUMINIO	237
k_values = [237];
c_values = [0];
##EN EL EXPERIMENTO DEL PROFE LA SUR ESTE Y NORTE NEUMANN; OESTE DIRITLETCH
##EL SUR Y ESTE NO VARIA; EL NORTE Y OESTE SI VARIA
##tipo_borde=[Borde_sur,Borde_este,Borde_norte,Borde_oeste]
#EXPRIENCIA INICIAL
#tipo_borde=[2,2,2,1];
#Expreciencia conmutar este oeste
#tipo_borde=[2,1,2,2];
#Expreciencia conmutar norte sur
#tipo_borde=[2,2,2,1];
#Expreciencia conmutar NS_EO
tipo_borde=[1,1,1,1];
#EXPRIENCIA INICIAL
% Número de nodos en el sur
valores_sur = [20, 50];
% Número de nodos en el este
valores_este = [20, 50];
#Expreciencia conmutar este oeste
##valores_este = [0:5:30];
#EXORIENCIA INICIAL
#valores_norte = [0:5:30];
#Expreciencia conmutar norte sur
valores_norte = [0:10:100];
#EXPERIENCIA INICIAL
#valores_oeste = [0:5:30];
#Expreciencia conmutar este oeste
valores_oeste = [0:10:100];
#EXPERIENCIA INICIAL

hay_condicion_ROB = ismember(3, tipo_borde);
hay_condicion_NEU = ismember(2, tipo_borde);
hay_condicion_DIR = ismember(1, tipo_borde);

DIR=[];
NEU=[];
ROB=[];

cant_total=length(valores_sur)*length(valores_norte)*length(valores_este)*length(valores_oeste)*length(G_values)*length(k_values)*length(c_values)
dataset_matriz=zeros(cant_total,7,7,20);
display(size(dataset_matriz))
contador=1;

##REcorremos todos los nodos asi determinar cual es interior y cual frontera
##RECORRA TODOS LOS NODOS Y LAS FRONTERAS
for i = 1:length(xnode(:,1))
##  is_present_NEU = 0;
##  is_present_DIR = 0;
##  is_present_NEU = ismember(i, NEU(:, 1));
##  columna_NEU(i,1)=is_present_NEU;
##  is_present_DIR = ismember(i, DIR(:, 1));
##  columna_DIR(i,1)=is_present_DIR;
##  es_borde=or(is_present_NEU,is_present_DIR);
##  borde(i,:) = [not(es_borde) ; es_borde##  is_present_NEU = 0;
is_present_al_SUR = 0;
is_present_al_ESTE = 0;
is_present_al_NORTE = 0;
is_present_al_OESTE = 0;

is_present_al_SUR = ismember(i, nodos_Sur);
is_present_al_ESTE = ismember(i, nodos_Este);
is_present_al_NORTE = ismember(i, nodos_Norte);
is_present_al_OESTE = ismember(i, nodos_Oeste);

es_borde=or(is_present_al_SUR,is_present_al_ESTE,is_present_al_NORTE,is_present_al_OESTE);
borde(i,1) = not(es_borde);
borde(i,2) = es_borde;
end

##for i = 1:length(DIR_values)
##    for j = 1:length(NEU_values)

for i=1:length(valores_sur)
                if (tipo_borde(1,1)==1)
                  % Función coseno para valores_sur
                  valores_DIR = (cos(linspace(0, 2*pi, 7)) * valores_sur(i));
                  DIR=[DIR;nodos_Sur,valores_DIR'];
                endif
                  if (tipo_borde(1,1)==2)
                  valores_NEU=ones(length(nodos_Sur),1)*valores_sur(i);
                  direccion_flujo=ones(length(nodos_Sur),1);
                  NEU=[NEU;nodos_Sur,valores_NEU,direccion_flujo];
                endif
                  if (tipo_borde(1,1)==3)

                endif


  for j=1:length(valores_este)

                if (tipo_borde(1,2)==1)
                  % Función cuadrática para valores_este
                  valores_DIR = (linspace(-1, 1, 7).^2) * valores_este(i);
                  DIR=[DIR;nodos_Este,valores_DIR'];
                endif
                  if (tipo_borde(1,2)==2)
                  valores_NEU=ones(length(nodos_Este),1)*valores_este(j);
                  direccion_flujo=ones(length(nodos_Este),1)*2;
                  NEU=[NEU;nodos_Este,valores_NEU,direccion_flujo];
                endif
                  if (tipo_borde(1,2)==3)

                endif
    for o=1:length(valores_norte)

                if (tipo_borde(1,3)==1)
                  valores_DIR=ones(length(nodos_Norte),1)*valores_norte(o);
                  DIR=[DIR;nodos_Norte,valores_DIR];
                endif
                  if (tipo_borde(1,3)==2)
                  valores_NEU=ones(length(nodos_Norte),1)*valores_norte(o);
                  direccion_flujo=ones(length(nodos_Norte),1)*3;
                  NEU=[NEU;nodos_Norte,valores_NEU,direccion_flujo];
                endif
                  if (tipo_borde(1,3)==3)

                endif
      for p=1:length(valores_oeste)
                if (tipo_borde(1,4)==1)
                  valores_DIR=ones(length(nodos_Oeste),1)*valores_oeste(p);
                  DIR=[DIR;nodos_Oeste,valores_DIR];
                endif
                  if (tipo_borde(1,4)==2)
                  valores_NEU=ones(length(nodos_Oeste),1)*valores_oeste(p);
                  direccion_flujo=ones(length(nodos_Oeste),1)*4;
                  NEU=[NEU;nodos_Oeste,valores_NEU,direccion_flujo];
                endif
                  if (tipo_borde(1,4)==3)

                endif
            for l = 1:length(k_values)
              for k = 1:length(G_values)
                for m = 1:length(c_values)
##                    DIR(:,2) = ones(size(DIR,1),1)*DIR_values(i);
##                    NEU(:, 2) = ones(size(NEU,1),1)*NEU_values(j);
##                    ROB = ROB_values;
## DIR=DIR
## NEU=NEU
                    model.G = [ones(size(model.G,1),1)*G_values(k)];
                    model.c = [ones(size(model.G,1),1)*c_values(m)];
                    model.k = [ones(size(model.G,1),1)*k_values(l)];
                    [PHI_temp, Q_temp] = main_proyecto(xnode, icone, DIR, NEU, ROB, model.G, model.k, model.c);
                    #display("caso calculado")
                    ##TEnemos que controlar si hay una condicion diritletch
                  if(hay_condicion_DIR)
                        DIR_indices = DIR(:,1);
                        DIR_values = DIR(:,2);
                        columna_DIR(DIR_indices,2) = DIR_values;
                        is_present_nodo_alnorte = ismember(DIR_indices, nodos_Norte);
                        is_present_nodo_aloeste = ismember(DIR_indices, nodos_Oeste);
                        is_present_nodo_aleste = ismember(DIR_indices, nodos_Este);
                        is_present_nodo_alsur = ismember(DIR_indices, nodos_Sur);
                        mascara_tipos_bordes_dataset(DIR_indices(is_present_nodo_alnorte),3) = 1;
                        mascara_tipos_bordes_dataset(DIR_indices(is_present_nodo_aloeste),4) = 1;
                        mascara_tipos_bordes_dataset(DIR_indices(is_present_nodo_aleste),2) = 1;
                        mascara_tipos_bordes_dataset(DIR_indices(is_present_nodo_alsur),1) = 1;
                    endif
                    if(hay_condicion_NEU)
                        NEU_indices = NEU(:,1);
                        NEU_values = NEU(:,2);
                        columna_NEU(NEU_indices,2) = NEU_values;
                        mascara_tipos_bordes_dataset(sub2ind(size(mascara_tipos_bordes_dataset), NEU_indices, NEU(:,3))) = 1;
                    endif
##          dataset = [xnode,borde,columna_DIR,columna_NEU];
##xnode[x,y] borde[interior, frontera] mascarara_tipos[norte, sur, este, oeste] no se si es ese orde
                    salida_vector=[xnode,borde,model.k,model.G,model.c,mascara_tipos_bordes_dataset,columna_DIR,columna_NEU,columna_ROB,PHI_temp,Q_temp];
                    dataset = [dataset;salida_vector];
                    ##dataset_matriz=[dataset_matriz;reordenar_a_matriz(salida_vector,7)];
                    dataset_matriz(contador,:,:,:)=[reordenar_a_matriz(salida_vector,7)];
                    contador=contador+1;

##                    salida_vector_un_elemento=full(dataset(1,:))'
##                    salida_matriz_un_elemento=full(dataset_matriz(1,1,:))
                endfor
            endfor
##        end
##    end
##    display("salio del for mas lejano")
##end
        endfor
      endfor
    endfor
      display("for este")
  endfor
  display("for sur")
endfor
#dataset_matriz=dataset_matriz
save("-v6",'mi_matriz_solo_diritletch_enriquesida.mat', 'dataset_matriz')
save("-v6",'mi_vector_solo_diritletch_enriquesida.mat', 'dataset')
#csvwrite("datos_sinteticos_co.csv",dataset)
#csvwrite("datos_sinteticos_matriz_ver_7.csv",dataset)



