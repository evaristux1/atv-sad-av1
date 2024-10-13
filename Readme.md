# Relatório Explicativo

## 1. Introdução

O Exame Nacional do Ensino Médio (ENEM) é uma das principais avaliações educacionais no Brasil, utilizado como critério de acesso ao ensino superior. Este projeto tem como objetivo desenvolver um modelo de previsão de desempenho no ENEM utilizando o algoritmo k-Nearest Neighbors (k-NN). Através da análise de microdados do ENEM, buscamos identificar características que influenciam o desempenho dos alunos e fornecer previsões para novos candidatos.

## 2. Manipulação dos Dados

### 2.1. Coleta de Dados

Os dados utilizados para este projeto foram coletados dos **Microdados do ENEM** disponibilizados pelo INEP. O conjunto de dados contém informações detalhadas sobre os participantes, incluindo notas, características socioeconômicas e dados sobre a escola de origem.

### 2.2. Preparação dos Dados

Os dados foram carregados utilizando a biblioteca `pandas`, e as seguintes etapas de preparação foram realizadas:

1. **Filtragem de Colunas**: Selecionamos as colunas relevantes para a previsão, que incluem:
   - Notas nas áreas do conhecimento (Matemática, Ciências da Natureza, Linguagens, Ciências Humanas, Redação).
   - Informações socioeconômicas (renda familiar, escolaridade da mãe, tipo de escola, raça/cor).

2. **Tratamento de Valores Faltantes**: Os registros com valores ausentes nas colunas relevantes foram removidos para garantir a qualidade dos dados.

3. **Codificação de Variáveis Categóricas**: Utilizamos o `LabelEncoder` para transformar variáveis categóricas em valores numéricos. Por exemplo, as variáveis para escolaridade da mãe e tipo de escola foram convertidas em inteiros, facilitando a análise.

4. **Normalização**: Os dados de renda familiar foram mantidos como valores `float` para melhor representação.

### 2.3. Conjunto de Treinamento e Teste

Os dados foram divididos em conjuntos de treinamento e teste utilizando a função `train_test_split` da biblioteca `sklearn`. O conjunto de treinamento foi usado para treinar o modelo, enquanto o conjunto de teste permitiu avaliar a precisão das previsões.

## 3. Implementação do Algoritmo

### 3.1. Escolha do Algoritmo

Optamos por utilizar o algoritmo k-Nearest Neighbors (k-NN) devido à sua simplicidade e eficácia em problemas de previsão baseados em dados históricos. O k-NN funciona identificando os k vizinhos mais próximos de uma nova entrada e fazendo previsões com base nas características desses vizinhos.

### 3.2. Treinamento do Modelo

O modelo k-NN foi implementado utilizando a classe `KNeighborsRegressor` da biblioteca `sklearn`. O processo de treinamento incluiu:

1. **Instanciação do Modelo**: Criamos uma instância do `KNeighborsRegressor` com um número definido de vizinhos (k = 5).
2. **Treinamento**: O modelo foi treinado utilizando os dados de entrada (X) e as respectivas notas (y).

### 3.3. Avaliação do Modelo

Após o treinamento, o modelo foi avaliado utilizando o conjunto de teste. O erro quadrático médio (MSE) foi calculado para medir a precisão das previsões. Valores de MSE mais baixos indicam um modelo mais preciso.

## 4. Justificativa para as Previsões

As previsões geradas pelo modelo k-NN são baseadas nas características dos alunos que participaram do ENEM em anos anteriores. Ao utilizar variáveis como renda familiar, escolaridade da mãe, tipo de escola e raça/cor, o modelo consegue identificar padrões que estão associados a um melhor desempenho no exame.

Os dados foram escolhidos com base em sua relevância para o desempenho acadêmico, e a normalização das entradas garante que o modelo possa generalizar bem para novos alunos. As previsões são apresentadas de forma clara e informativa, permitindo que a instituição de ensino identifique quais alunos podem precisar de apoio adicional.

## 5. Conclusão

Este projeto demonstrou como utilizar aprendizado de máquina para prever o desempenho no ENEM com base em microdados disponíveis. A implementação do algoritmo k-NN, juntamente com a manipulação cuidadosa dos dados, resultou em um modelo capaz de fornecer previsões valiosas. Futuras iterações podem incluir a adição de mais variáveis e a exploração de algoritmos mais complexos para melhorar ainda mais a precisão das previsões.
