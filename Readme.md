
## Documentação do Código para Previsão de Desempenho no ENEM com k-NN

### Descrição Geral

Este código utiliza o algoritmo k-Nearest Neighbors (k-NN) para prever as notas do Exame Nacional do Ensino Médio (ENEM) de estudantes com base em fatores socioeconômicos e históricos de notas anteriores. O fluxo principal envolve o carregamento de dados, pré-processamento, treinamento de um modelo de regressão k-NN e a realização de previsões.

---

### Bibliotecas Utilizadas
- **pandas**: Manipulação de dados em formato de DataFrame.
- **numpy**: Operações matemáticas e manipulação de arrays.
- **scikit-learn**: Biblioteca para aprendizado de máquina, usada para treinar o modelo k-NN.
- **joblib**: Serialização do modelo para salvamento e carregamento posterior.

---

### Funções Implementadas

1. **`load_principal_columns(input_file_path, output_file_path)`**
   - **Descrição**: Carrega um arquivo CSV, seleciona as colunas relevantes para o modelo e salva um novo arquivo CSV com essas colunas.
   - **Parâmetros**:
     - `input_file_path` (str): Caminho para o arquivo CSV de entrada.
     - `output_file_path` (str): Caminho para salvar o novo CSV com as colunas filtradas.
   - **Retorno**: Nenhum. O novo arquivo CSV é salvo no caminho especificado.

2. **`encode_categorical_columns(df)`**
   - **Descrição**: Codifica variáveis categóricas e trata valores ausentes nas colunas relevantes.
   - **Parâmetros**:
     - `df` (DataFrame): DataFrame a ser tratado.
   - **Retorno**: DataFrame tratado com colunas categóricas codificadas e valores nulos preenchidos.

3. **`load_and_prepare_data(file_path)`**
   - **Descrição**: Carrega o arquivo CSV, filtra colunas relevantes e trata valores ausentes.
   - **Parâmetros**:
     - `file_path` (str): Caminho do arquivo CSV a ser carregado.
   - **Retorno**: DataFrame tratado com colunas filtradas.

4. **`train_and_save_knn_model(df, model_path="modelo_knn.pkl")`**
   - **Descrição**: Treina um modelo de regressão k-NN com base nos dados fornecidos e salva o modelo.
   - **Parâmetros**:
     - `df` (DataFrame): Dados de treinamento.
     - `model_path` (str): Caminho onde o modelo será salvo.
   - **Retorno**: Nenhum. O modelo treinado é salvo no caminho especificado.

5. **`load_knn_model(model_path="modelo_knn.pkl")`**
   - **Descrição**: Carrega um modelo k-NN previamente salvo.
   - **Parâmetros**:
     - `model_path` (str): Caminho para o arquivo do modelo salvo.
   - **Retorno**: Objeto do modelo k-NN carregado ou `None` se o modelo não for encontrado.

6. **`ask_escolaridade_mae()`**
   - **Descrição**: Solicita e retorna a escolaridade da mãe do estudante com base em uma entrada do usuário.
   - **Retorno**: Código correspondente à escolaridade da mãe.

7. **`ask_etnia()`**
   - **Descrição**: Solicita e retorna a etnia do estudante com base em uma entrada do usuário.
   - **Retorno**: Código numérico correspondente à etnia.

8. **`ask_float_input(prompt)`**
   - **Descrição**: Valida uma entrada de número decimal do usuário.
   - **Parâmetros**:
     - `prompt` (str): Mensagem de solicitação.
   - **Retorno**: Valor decimal inserido pelo usuário.

9. **`ask_int_input(prompt, valid_values=None)`**
   - **Descrição**: Valida uma entrada de número inteiro do usuário.
   - **Parâmetros**:
     - `prompt` (str): Mensagem de solicitação.
     - `valid_values` (list): Lista opcional de valores válidos.
   - **Retorno**: Valor inteiro inserido pelo usuário.

10. **`get_renda_codigo(renda_familiar)`**
    - **Descrição**: Converte um valor de renda familiar em seu respectivo código (Q006).
    - **Parâmetros**:
      - `renda_familiar` (float): Valor da renda familiar.
    - **Retorno**: Código correspondente à renda.

11. **`ask_uf()`**
    - **Descrição**: Solicita e retorna o código do Estado (UF) do estudante com base em uma entrada do usuário.
    - **Retorno**: Valor codificado do estado (UF).

12. **`ask_municipio()`**
    - **Descrição**: Solicita e retorna o código do município da prova com base em uma entrada do usuário.
    - **Retorno**: Código do município.

13. **`ask_simulado_scores()`**
    - **Descrição**: Solicita e retorna as notas do simulado nas diferentes áreas do conhecimento.
    - **Retorno**: Dicionário contendo as notas nas diferentes áreas do conhecimento.

14. **`give_financial_value()`**
    - **Descrição**: Solicita o valor da renda familiar e retorna o código codificado de renda (Q006).
    - **Retorno**: Valor codificado da renda familiar.

15. **`predict_new_student(knn)`**
    - **Descrição**: Solicita os dados de um novo aluno e realiza a previsão das notas do ENEM utilizando o modelo k-NN.
    - **Parâmetros**:
      - `knn` (KNeighborsRegressor): Modelo k-NN treinado.
    - **Retorno**: Nenhum. As previsões são exibidas.

---

### Fluxo Geral

1. **Carregar Dados**: Utilize a função `load_principal_columns` para carregar as colunas relevantes dos dados do ENEM e salvar um CSV otimizado.
2. **Pré-processamento**: O DataFrame resultante passa pela função `encode_categorical_columns` para codificação das variáveis categóricas e tratamento de valores ausentes.
3. **Treinamento do Modelo**: A função `train_and_save_knn_model` é usada para treinar o modelo k-NN com os dados pré-processados e salvar o modelo treinado.
4. **Predição**: Novos dados de estudantes podem ser passados para o modelo através da função `predict_new_student` para prever as notas com base nos dados inseridos.
