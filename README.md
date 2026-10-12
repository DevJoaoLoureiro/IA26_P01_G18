PROJETO 1 — INTELIGÊNCIA ARTIFICIAL (2026/2027)
Grupo 18 — IA26_P01_G18

TEMA:
Otimização de Rotas para Distribuição de Alimentos Locais

REPOSITÓRIO GITHUB:
https://github.com/DevJoaoLoureiro/IA26_P01_G18

DESCRIÇÃO:
Este projeto tem como objetivo desenvolver um agente de Inteligência Artificial capaz de planear e otimizar rotas de recolha e entrega de produtos alimentares locais.

O problema considera dois veículos, produtores, clientes e locais de entrega, procurando minimizar a distância total percorrida e respeitar as restrições de capacidade dos veículos, precedência entre recolhas e entregas e janelas temporais.

ALGORITMOS IMPLEMENTADOS:
- Greedy construtivo, com duas heurísticas.
- Greedy Best-First Search (GBFS), com duas heurísticas.
- A*, com duas heurísticas.
- Hill Climbing, para otimização local de soluções completas.
- Constraint Satisfaction Problem (CSP), para obtenção de soluções viáveis.

ESTRUTURA DO PROJETO:
- 01_eda_tiny.ipynb — Análise Exploratória dos Dados.
- 02_relatorio.ipynb — Relatório, execução experimental, gráficos e conclusões.
- algoritmos.py — Modelação do problema e implementação dos algoritmos.
- data_tiny/ — Dataset utilizado, constituído por 11 ficheiros CSV.
- readme.txt — Descrição do projeto e ligação ao repositório.

EXECUÇÃO:
1. Abrir o notebook 02_relatorio.ipynb no Google Colab.
2. Garantir acesso ao repositório privado do GitHub e configurar o segredo GITHUB_TOKEN no Colab, caso se utilize a importação automática do repositório.
3. Executar as células do notebook pela ordem apresentada.
4. Consultar os resultados experimentais dos cenários com 3, 5 e 11 artigos.

Os resultados incluem tabelas comparativas, tempos de execução, número de nós expandidos, distâncias percorridas e gráficos de evolução do Hill Climbing.

NOTEBOOKS NO GOOGLE COLAB:
EDA:
https://colab.research.google.com/drive/12FWaIvBdJSoylMhv3waKl_61A0MqUpJT

RELATÓRIO:
https://colab.research.google.com/drive/1weJqQm-Yb1vAAoeTRzOOaSqEdxYfGAa-

NOTAS:
- O dataset utilizado é o data_tiny.
- Os algoritmos de pesquisa utilizam limites computacionais.
- Não é garantida a otimalidade global das soluções obtidas.
- O Hill Climbing utiliza uma solução completa da GBFS quando disponível; caso contrário, pode utilizar uma solução do CSP.