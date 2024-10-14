### Relatório Explicativo: Previsão de Desempenho no ENEM usando k-Nearest Neighbors (k-NN)

#### 1. Introdução

Este projeto foi desenvolvido como parte da disciplina **Sistemas de Apoio à Tomada de Decisão (SAD)**, com o objetivo de prever o desempenho de futuros candidatos no Exame Nacional do Ensino Médio (ENEM). Para isso, utilizamos o algoritmo **k-Nearest Neighbors (k-NN)**, que prevê as notas de um novo aluno com base em suas características pessoais e histórico escolar, comparando-o com dados de alunos de anos anteriores.

A seguir, detalhamos o processo de manipulação dos dados, implementação do algoritmo e justificativas para as previsões realizadas.

#### 2. Manipulação dos Dados

##### 2.1. Coleta e Pré-processamento

Os dados utilizados são provenientes dos **Microdados do ENEM**, que contêm informações detalhadas dos participantes, como notas em áreas de conhecimento e dados socioeconômicos. As variáveis selecionadas foram:

- **Notas nas Áreas do Conhecimento**:
  - Matemática (NU_NOTA_MT)
  - Ciências da Natureza (NU_NOTA_CN)
  - Linguagens (NU_NOTA_LC)
  - Ciências Humanas (NU_NOTA_CH)
  - Redação (NU_NOTA_REDACAO)
  
- **Informações Socioeconômicas**:
  - Renda Familiar (Q006)
  - Escolaridade da Mãe (Q002)
  - Tipo de Escola (Pública ou Privada - TP_ESCOLA)
  - Raça/Cor (TP_COR_RACA)

- **Localidade**:
  - Estado da Prova (SG_UF_PROVA)
  - Município da Prova (CO_MUNICIPIO_PROVA)

##### 2.2. Limpeza dos Dados

Após carregar os dados, foi necessário remover entradas com valores ausentes nas colunas relevantes, utilizando a função `dropna()`. Além disso, as variáveis categóricas (como renda familiar e tipo de escola) foram convertidas para valores numéricos usando a técnica de **label encoding**. Essa codificação permitiu que o algoritmo k-NN, que depende de distâncias entre pontos, funcionasse corretamente.

##### 2.3. Normalização dos Dados

Uma vez que as variáveis apresentavam escalas diferentes (por exemplo, notas variando de 0 a 1000 e renda familiar em faixas), foi necessário normalizar as variáveis para garantir que uma não dominasse a outra no cálculo das distâncias. A normalização foi feita usando técnicas de **escalonamento entre 0 e 1**.

#### 3. Implementação do Algoritmo k-NN

##### 3.1. Algoritmo k-NN

O k-NN é um algoritmo de aprendizado supervisionado que classifica ou faz previsões com base nos "k" vizinhos mais próximos de um ponto de dados. Para prever as notas do ENEM de novos alunos, o algoritmo compara suas características com as de alunos cujos dados já são conhecidos, utilizando a **distância euclidiana**.

##### 3.2. Divisão de Dados

Os dados foram divididos em conjuntos de **treinamento (80%)** e **teste (20%)** utilizando a função `train_test_split`. O conjunto de treinamento foi utilizado para ajustar o modelo, enquanto o conjunto de teste serviu para avaliar a capacidade preditiva do modelo.

##### 3.3. Valor de k

Testamos diferentes valores de k (como 3, 5, 7 e 10) e, através de validação cruzada, identificamos que **k=10** proporcionou o melhor desempenho, equilibrando precisão e capacidade de generalização.

##### 3.4. Métrica de Avaliação

O modelo foi avaliado utilizando o **Erro Quadrático Médio (MSE)**, uma métrica comum para modelos de regressão que quantifica a diferença média entre os valores previstos e os reais. Quanto menor o MSE, melhor o modelo está ajustado.

#### 4. Justificativa das Previsões

As previsões do modelo são baseadas no princípio de que alunos com características socioeconômicas e educacionais semelhantes tendem a ter desempenhos parecidos no ENEM. O algoritmo k-NN identifica os vizinhos mais próximos do novo aluno e faz uma média ponderada das notas desses vizinhos para gerar a previsão.

Por exemplo, um aluno com renda familiar e escolaridade dos pais em níveis médios terá sua previsão baseada em alunos históricos com características semelhantes. O modelo não só prevê as notas em cada área do conhecimento, mas também compara as previsões com as notas reais dos alunos mais próximos, fornecendo uma **justificativa baseada em dados históricos**.

#### 5. Resultados e Análise

Durante a fase de teste, o modelo apresentou um MSE relativamente baixo, especialmente para as áreas de **Matemática** e **Ciências da Natureza**, onde os padrões de desempenho são mais bem definidos. As previsões para áreas como **Redação** e **Linguagens** foram mais variáveis, possivelmente devido à maior subjetividade na correção dessas provas e à influência de fatores contextuais.

A principal limitação identificada foi a dependência dos dados disponíveis. Em alguns estados ou municípios com menos alunos registrados, as previsões podem ser menos precisas. Além disso, a ausência de dados adicionais, como a qualidade das escolas, pode ter limitado a capacidade preditiva do modelo.

#### 6. Conclusão

O uso do algoritmo k-NN para prever o desempenho no ENEM demonstrou ser eficaz, especialmente para alunos com características mais comuns. O modelo é capaz de identificar padrões a partir de dados históricos e utilizá-los para fazer previsões confiáveis sobre o desempenho de novos alunos.

Recomendamos, para futuras melhorias, a incorporação de mais variáveis socioeconômicas e a experimentação com algoritmos mais complexos, como **redes neurais**, para refinar ainda mais as previsões.
