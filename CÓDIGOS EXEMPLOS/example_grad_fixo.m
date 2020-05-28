%**********************************************************
% Disciplina Aprendizado de M�quina
% Exemplo Algoritmo do Gradiente Descente com passo fixo
% Problema de otimiza��o
%               min f(x(1),x(2))= (x(1)-3)^2 + (x(2)-2)^2
% Exemplo
%        example_grad - n�o precisa definir o valor de x
%        exampe_grad([0 0]) - ponto inicial x =[0 0]
%**********************************************************

function example_grad_fixo(x)
% Defini��o de par�metros
itmax = 1000;     % Numero m�ximo de itera��es
normagrad = 1e-3; % Norma m�nima do gradiente
alfa = 0.1;       % taxa de aprendizado
%Valor inicial para x
if nargout<1
    x = rands(2,1);
end
% Avalia a fun��o e c�lcula o gradiente
[f,g,~]=calc_derivada(x);
% Defini��o do valor itera��o inicial
it = 0;
% Verifica crit�rio de parada (numero de itera��es ou norma do gradiente)
while it<itmax & norm(g)>normagrad
    % Imprime na tela
    fprintf('Itera��es = %d, norma do gradiente =%1.4f,  x = [%2.5f %2.5f] \n',it,norm(g),x(1),x(2))
    %Incrementa o numero de itera��es
    it = it + 1;
    % Atualiza
    x = x - alfa*g;
    % Avalia a fun��o e c�lcula do gradiente
    [f,g,~]=calc_derivada(x);
end


function [f,g,h]=calc_derivada(x)
f = (x(1)-3)^2+(x(2)-2)^2;
g = [2*(x(1)-3);2*(x(2)-2)];
h = [2 0;0 2];