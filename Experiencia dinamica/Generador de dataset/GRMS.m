function [frecpts grms]=GRMS(nomarch,Nfiltros,eje)

%frecpts=puntos de frecuencias en los que se colocaran los valores de GRMS
%grms=gRMS resultantes para cada punto de frecuencia

%nomarch=Nombre del archivo, por defecto debe ser 'aceleraciones.txt'
%Nfiltros=cantidad de divisiones del espectro de frecuencia
%eje='x', 'y' o 'z'

	x=load(nomarch);

	t=x(1:end,1);

	if(eje=='x')
		x=x(1:end,2);
	elseif(eje=='y')
		x=x(1:end,3);
	elseif(eje=='z')
		x=x(1:end,4);
	else
		x=x(1:end,2);
	end



	%t=tiempo';
	%x=3*sin(4*pi*t) + 20*sin(12*pi*t) + 15*sin(20*pi*t);


	dt=t(2)-t(1)
  %dt=0.0051; % valor promediado
	fs=round(1/dt);
	N=length(t);
	df=fs/N;

	%dividimos el espectro en 3 fi-f1 / f1-f2 / f2-ff
	%las frecuencias resonantes son las que van desde 1.2385 2.2408 por lo que las vemos en:

	f=zeros(1,Nfiltros);
	frecpts=zeros(1,Nfiltros);
	y=zeros(Nfiltros,N);
	%figure

	dfs=fs/Nfiltros; %diferencial de frecuencias de muestreo

	i=1;
	grms=zeros(1,Nfiltros);
	suma_rms=0;
	for fi=dfs:dfs:Nfiltros*dfs %

		f(i)=fi;
		if(i==1)
			f0=0;
		else
			f0=f(i-1);
		end


		frecpts(i)=(f0+f(i))/2;

		%continue;
		%printf('Filtro... ');
		y(i,1:end)=filtro(x,f0,f(i),fs,i,Nfiltros);
		%printf('Listo!\nGRMS... ');
       % fprintf('\nSalio del filtro');
	   rms_ = rms2(y(i));
	   %fprintf('RMS %i: %f',[i rms_]);
	   suma_rms=suma_rms+rms_;
	   grms(i)=(rms_^2)/dfs;
       % fprintf('\nG');
		%printf('Listo!');
		i=i+1;
      %  fprintf('Valor de i=%i',i);

	end



	fprintf('RMS calculado: %f - RMS Total: %f',suma_rms,rms2(x));


	figure
	plot(frecpts,grms);
    grid on
    grid minor
	title('GRMS');
	xlabel('Frecuencias');
	ylabel('|GRMS|');


end

function [out] = rms2(x)
%RMS Compute the Root Mean Square value of the input vector
% This function computes the RMS value using a bit more of a brute-force
% method.

	out = sqrt(sum((x.^2)/max(size(x))));

end


function [y_filtrada]=filtro(x,fc1,fc2,fs,iplot,Nplot)

	if(fc1>fc2)
		y_filtrada=[];
		return;
	end



	N=length(x);
	%amp=sqrt(sum(x.^2))/N;
	df=fs/N;

	f1=[round(fc1/df/2),round(fc2/df/2)];
	f2=[round(fs/df) round(fs/df)]-f1;


	H=zeros(1,N);
	if(f1(1)==0)
		f1(1)=1;
	end
	if(f2(2)==0)
		f2(2)=1;
	end


	H(1,f1(1):f1(2))=1;
    H(1,f2(2):f2(1))=1;

    if(fc1==0)
        H(1)=0;
		H(2)=0;
        H(N)=0;
		H(N-1)=0;
    end
	%fprintf('Algo');
	frec=[0:df:fs-df];

% 	subplot(Nplot,1,iplot);
% 	stem(frec,H);
%
%
% 	title('Filtro');
% 	xlabel('Frecuencias');
% 	ylabel('|X|');

	h=ifft(H);
	vent=blackman(N);

	mid=floor(N/2);
    %fprintf('otro');
	h=[h(mid+1:end),h(1:mid)];

	h=vent'.*h;
	%fprintf('Final');

	%H=abs(ifft(h));
	%X=abs(ifft(x));

	y_filtrada=conv(real(x),real(h));
   % fprintf('Conv');

	y_filtrada=y_filtrada(mid+1:mid+N);
   % fprintf('Filtrada');
	%length(y_filtrada)


	%Y=X.*H;

	%y_filtrada=real(ifft(Y));
end
