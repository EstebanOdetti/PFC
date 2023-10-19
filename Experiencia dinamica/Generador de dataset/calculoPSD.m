% carga de datos para obtener PSD
% calculoPSD.m
clear all; clc; clf

data = load('RuedaDel.txt');
time=data(:,1);
ax=data(:,2);
ay=data(:,3);
az=data(:,4);
resul = [time ax ay az];
size(resul)
my_fprintf('RuedaDel.txt','%12.7f %12.15f %12.15f %12.15f \n',resul);
[frecptsx grmsx]=GRMS('RuedaDel.txt',30,'x')
[frecptsy grmsy]=GRMS('RuedaDel.txt',30,'y')
[frecptsz grmsz]=GRMS('RuedaDel.txt',30,'z')

acelx=[frecptsx' grmsx'];
acely=[frecptsy' grmsy'];
acelz=[frecptsz' grmsz'];

my_fprintf('PSDxB.txt','%12.7f %12.15f \n',acelx);
my_fprintf('PSDyB.txt','%12.7f %12.15f \n',acely);
my_fprintf('PSDzB.txt','%12.7f %12.15f \n',acelz);
