%**********************************************************
% Disciplina Aprendizado de Máquina
% Exemplo Algoritmo do Gradiente Descente com passo variável
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
%Valor inicial para x
if nargout<1
    x = rands(2,1);
end
% Avalia a função e cálcula o gradiente
[f,g,~]=calc_derivada(x);
% Definição do valor iteração inicial
it = 0;
% Imprime na tela
fprintf('Iterações = %d, norma do gradiente =%1.4f, alfa = %1.2f, x = [%2.5f %2.5f] \n',it,norm(g),0,x(1),x(2))
% Verifica critério de parada (numero de iterações ou norma do gradiente)
while it<itmax & norm(g)>normagrad
    %Incrementa o numero de iterações
    it = it + 1;
    % Cálculo do valor de alfa
    alfa = calc_alfa(x,-g);
    % Atualiza o valor de x
    x = x - alfa*g;
    % Avalia a função e cálcula do gradiente
    [f,g,~]=calc_derivada(x);
    % Imprime na tela
    fprintf('Iterações = %d, norma do gradiente =%1.4f, alfa = %1.2f, x = [%2.5f %2.5f] \n',it,norm(g),alfa,x(1),x(2))

end
end

function alfa = calc_alfa(x,dir)
epsilon = 1e-3;
hlmin = 1e-3;
alfa_l = 0;
alfa_u = rand();
xn = x + alfa_u*dir;
[f,g,~]=calc_derivada(xn);
hl = g'*dir;
while hl<0
    alfa_u = 2*alfa_u;
    xn = x + alfa_u*dir;
    [f,g,~]=calc_derivada(xn);
    hl = g'*dir;
end
alfa_m = (alfa_l+alfa_u)/2;
itmax = ceil(log ((alfa_u-alfa_l)/epsilon));
it = 0;
while abs(hl)>hlmin & it<itmax
    xn = x + alfa_m*dir;
    [f,g,~]=calc_derivada(xn);
    hl = g'*dir;
    if hl>0
        alfa_u = alfa_m;
    elseif hl<0
        alfa_l = alfa_m;
    else
        break;
    end
    alfa_m = (alfa_l+alfa_u)/2;
   
end
alfa = alfa_m;
end
    
function [f,g,h]=calc_derivada(x)
f = (x(1)-3)^2+(x(2)-2)^2;
g = [2*(x(1)-3);2*(x(2)-2)];
h = [2 0;0 2];
end