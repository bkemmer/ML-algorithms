%**********************************************************
% Disciplina Aprendizado de Máquina
% Exemplo Algoritmo do Gradiente Descente com passo fixo
% Problema de otimização
%               min f(x(1),x(2))= (x(1)-3)^2 + (x(2)-2)^2
% Exemplo
%        example_grad - não precisa definir o valor de x
%        exampe_grad([0 0]) - ponto inicial x =[0 0]
%**********************************************************

function example_grad_fixo(x)
% Definição de parâmetros
itmax = 1000;     % Numero máximo de iterações
normagrad = 1e-3; % Norma mínima do gradiente
alfa = 0.1;       % taxa de aprendizado
%Valor inicial para x
if nargout<1
    x = rands(2,1);
end
% Avalia a função e cálcula o gradiente
[f,g,~]=calc_derivada(x);
% Definição do valor iteração inicial
it = 0;
% Verifica critério de parada (numero de iterações ou norma do gradiente)
while it<itmax & norm(g)>normagrad
    % Imprime na tela
    fprintf('Iterações = %d, norma do gradiente =%1.4f,  x = [%2.5f %2.5f] \n',it,norm(g),x(1),x(2))
    %Incrementa o numero de iterações
    it = it + 1;
    % Atualiza
    x = x - alfa*g;
    % Avalia a função e cálcula do gradiente
    [f,g,~]=calc_derivada(x);
end


function [f,g,h]=calc_derivada(x)
f = (x(1)-3)^2+(x(2)-2)^2;
g = [2*(x(1)-3);2*(x(2)-2)];
h = [2 0;0 2];