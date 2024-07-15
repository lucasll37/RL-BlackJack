ATUALIZAR PARA AÇÃO-VALOR:



A função de valor \( V(s) \) em um \textit{MDP} é definida como a expectativa do retorno futuro a partir do estado \( s \), seguindo uma política \( \pi \):

\begin{equation}
    V_{\pi}(s) = \mathbb{E}_{\pi} [G_t | S_t = s]
\end{equation}

Onde \( G_t \) é o retorno acumulado a partir do tempo \( t \), dado por:

\begin{equation}
    G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}
\end{equation}

A função de valor pode ser estimada iterativamente usando a equação de Bellman:

\begin{equation}
    V_{\pi}(s) = \sum_{a \in A} \pi(a|s) \sum_{s' \in S} p(s'|s,a) \left[ r(s,a,s') + \gamma V_{\pi}(s') \right]
\end{equation}




COMO ESTABILIZAR OS PESOS DA REDE?


DESENVOLVIMENTO:

ALEM DE CITAR OS CÓDIGO FONTE NO APENDICE, FALAR COMO FOI FEITO O QUASI OPTIMUM MONTE CARLO


RESULTADO
INCLUIR NA TABELA O OPTIMUM MONTE CARLO



INCLUIR LINK DE REFERENCIAS


REformular Conclusão para que sejam explorados problemas mais desafiadores