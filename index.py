import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import joblib 

# Função para gerar um CSV com apenas as colunas relevantes afim de otmizar o processamento
def load_principal_columns(input_file_path, output_file_path):
    # Carregar o arquivo CSV completo
    df = pd.read_csv(input_file_path, sep=';', encoding='latin-1')

    colunas_relevantes = [
        'NU_NOTA_MT',     # Nota em Matemática e suas Tecnologias
        'NU_NOTA_CN',     # Nota em Ciências da Natureza e suas Tecnologias
        'NU_NOTA_LC',     # Nota em Linguagens, Códigos e suas Tecnologias
        'NU_NOTA_CH',     # Nota em Ciências Humanas e suas Tecnologias
        'NU_NOTA_REDACAO', # Nota da Redação
        'Q006',           # Renda Familiar
        'Q002',           # Escolaridade da Mãe
        'TP_ESCOLA',      # Tipo de Escola (Pública ou Privada)
        'TP_COR_RACA',    # Raça/Cor
        'SG_UF_PROVA',    # Estado da Prova
        'CO_MUNICIPIO_PROVA'  # Município da Prova
    ]

    # Filtrar as colunas relevantes e salvar em um novo arquivo CSV
    df_relevante = df[colunas_relevantes].dropna()
    
    # Salvar o DataFrame filtrado em um novo CSV
    df_relevante.to_csv(output_file_path, sep=';', index=False, encoding='latin-1')
    print(f"Novo arquivo CSV gerado com sucesso em: {output_file_path}")

# Função para codificar variáveis categóricas e tratar valores nulos
def encode_categorical_columns(df):
    le = LabelEncoder()
    
    # Tratar a renda familiar (Q006): preenchendo valores nulos com a moda
    df['Q006'] = df['Q006'].fillna(df['Q006'].mode()[0])  # Usar a moda como valor padrão
    df['Q006'] = le.fit_transform(df['Q006'].astype(str))  # Converter para valores numéricos
    
    # Tratar a escolaridade da mãe (Q002): preenchendo valores nulos com a moda
    df['Q002'] = df['Q002'].fillna(df['Q002'].mode()[0])
    df['Q002'] = le.fit_transform(df['Q002'].astype(str))  # Converter para valores numéricos

    # Tratar o tipo de escola (TP_ESCOLA): preenchendo valores nulos com 0 (padrão: pública)
    df['TP_ESCOLA'] = df['TP_ESCOLA'].fillna(0)  

    # Tratar a raça/cor (TP_COR_RACA): preenchendo valores nulos com 0 (padrão: não informado)
    df['TP_COR_RACA'] = df['TP_COR_RACA'].fillna(0)

    # Tratar UF (Estado da Prova): preenchendo valores nulos com a moda e aplicando LabelEncoder
    df['SG_UF_PROVA'] = df['SG_UF_PROVA'].fillna(df['SG_UF_PROVA'].mode()[0])
    df['SG_UF_PROVA'] = le.fit_transform(df['SG_UF_PROVA'].astype(str))  # Converte siglas de UF para numérico

    # Tratar o código do município (CO_MUNICIPIO_PROVA): preenchendo valores nulos com 0
    df['CO_MUNICIPIO_PROVA'] = df['CO_MUNICIPIO_PROVA'].fillna(0)

    return df

# Função para carregar e preparar os dados
def load_and_prepare_data(file_path):
    # Carregar o arquivo CSV
    df = pd.read_csv(file_path, sep=';', encoding='latin-1')  # Ajuste o caminho do arquivo
    
    # Definir as colunas relevantes
    colunas_relevantes = [
        'NU_NOTA_MT',     # Nota em Matemática e suas Tecnologias
        'NU_NOTA_CN',     # Nota em Ciências da Natureza e suas Tecnologias
        'NU_NOTA_LC',     # Nota em Linguagens, Códigos e suas Tecnologias
        'NU_NOTA_CH',     # Nota em Ciências Humanas e suas Tecnologias
        'NU_NOTA_REDACAO', # Nota da Redação
        'Q006',           # Renda Familiar
        'Q002',           # Escolaridade da Mãe
        'TP_ESCOLA',      # Tipo de Escola (Pública ou Privada)
        'TP_COR_RACA',    # Raça/Cor
        'SG_UF_PROVA',    # Estado da Prova
        'CO_MUNICIPIO_PROVA'  # Município da Prova
    ]
    
    # Filtrar as colunas relevantes e remover as linhas com valores ausentes
    df = df[colunas_relevantes].dropna()

    # Mapeamento de Tipo de Escola (1: Pública, 2: Privada)
    df['TP_ESCOLA'] = df['TP_ESCOLA'].map({1: 0, 2: 1})  # 0 = Pública, 1 = Privada

    # Preenchendo valores ausentes em TP_COR_RACA com 0 (pode ajustar conforme a necessidade)
    df['TP_COR_RACA'] = df['TP_COR_RACA'].fillna(0)

    # Retornar o DataFrame preparado
    return df

# Função para treinar e salvar o modelo k-NN
def train_and_save_knn_model(df, model_path="modelo_knn.pkl"):
    X = df[['Q006', 'Q002', 'TP_ESCOLA', 'TP_COR_RACA', 'SG_UF_PROVA', 'CO_MUNICIPIO_PROVA','NU_NOTA_MT', 'NU_NOTA_CN', 'NU_NOTA_LC', 'NU_NOTA_CH', 'NU_NOTA_REDACAO']].values
    y = df[['NU_NOTA_MT', 'NU_NOTA_CN', 'NU_NOTA_LC', 'NU_NOTA_CH', 'NU_NOTA_REDACAO']].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    knn = KNeighborsRegressor(n_neighbors=10)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Erro Quadrático Médio (MSE): {mse:.2f}")

    # Salvar o modelo treinado
    joblib.dump(knn, model_path)
    print(f"Modelo salvo em {model_path}")

# Função para carregar o modelo k-NN salvo
def load_knn_model(model_path="modelo_knn.pkl"):
    try:
        knn = joblib.load(model_path)
        print(f"Modelo carregado de {model_path}")
        return knn
    except FileNotFoundError:
        print(f"Modelo não encontrado em {model_path}. Aguarde enquanto o modelo é treinado.")
        return None

# Função para perguntar a escolaridade da mãe
def ask_escolaridade_mae():
    print("Escolaridade da Mãe (Q002):")
    print("A: Nunca estudou")
    print("B: Não completou a 4ª série/5º ano do Ensino Fundamental")
    print("C: Completou a 4ª série/5º ano, mas não completou o Ensino Fundamental")
    print("D: Completou o Ensino Fundamental, mas não completou o Ensino Médio")
    print("E: Completou o Ensino Médio, mas não completou o Ensino Superior")
    print("F: Completou o Ensino Superior, mas não completou a Pós-graduação")
    print("G: Completou a Pós-graduação")
    
    while True:
        escolaridade_mae = input("Escolha a letra correspondente à escolaridade da mãe: ").upper()
        mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}
        if escolaridade_mae in mapping:
            return mapping[escolaridade_mae]
        else:
            print("Entrada inválida. Escolha uma opção válida (A-G).")

# Função para perguntar a etnia
def ask_etnia():
    print("Raça/Cor (1 a 5):")
    print("1: Branca")
    print("2: Parda")
    print("3: Negra")
    print("4: Amarela")
    print("5: Indígena")
    
    while True:
        etnia = input("Escolha o número correspondente à raça/cor: ")
        if etnia in {'1', '2', '3', '4', '5'}:  
            return int(etnia)  # Retorna o valor como inteiro
        else:
            print("Entrada inválida. Escolha uma opção válida (1-5).")

# Função para validar input numérico
def ask_float_input(prompt):
    while True:
        try:
            value = float(input(prompt))
            return value
        except ValueError:
            print("Entrada inválida. Digite um número válido.")

# Função para validar input inteiro
def ask_int_input(prompt, valid_values=None):
    while True:
        try:
            value = int(input(prompt))
            if valid_values and value not in valid_values:
                print(f"Entrada inválida. Escolha entre {valid_values}.")
            else:
                return value
        except ValueError:
            print("Entrada inválida. Digite um número inteiro válido.")

# Função para converter a renda em um código Q006
def get_renda_codigo(renda_familiar):
    # Definir os intervalos de renda e seus códigos correspondentes
    if renda_familiar == 0:
        return 'A'
    elif 1 <= renda_familiar <= 1000:
        return 'B'
    elif 1001 <= renda_familiar <= 2000:
        return 'C'
    elif 2001 <= renda_familiar <= 3000:
        return 'D'
    elif 3001 <= renda_familiar <= 4000:
        return 'E'
    elif 4001 <= renda_familiar <= 5000:
        return 'F'
    elif 5001 <= renda_familiar <= 6000:
        return 'G'
    elif 6001 <= renda_familiar <= 7000:
        return 'H'
    elif 7001 <= renda_familiar <= 8000:
        return 'I'
    elif 8001 <= renda_familiar <= 9000:
        return 'J'
    elif 9001 <= renda_familiar <= 10000:
        return 'K'
    elif renda_familiar > 10000:
        return 'L'
    else:
        return 'M'  # Para não declarado ou valor inválido


# Função para perguntar o Estado (UF)
def ask_uf():
    print("Informe o Estado (UF):")
    estados = {
        'AC': 'Acre', 'AL': 'Alagoas', 'AP': 'Amapá', 'AM': 'Amazonas', 'BA': 'Bahia',
        'CE': 'Ceará', 'DF': 'Distrito Federal', 'ES': 'Espírito Santo', 'GO': 'Goiás',
        'MA': 'Maranhão', 'MT': 'Mato Grosso', 'MS': 'Mato Grosso do Sul', 'MG': 'Minas Gerais',
        'PA': 'Pará', 'PB': 'Paraíba', 'PR': 'Paraná', 'PE': 'Pernambuco', 'PI': 'Piauí',
        'RJ': 'Rio de Janeiro', 'RN': 'Rio Grande do Norte', 'RS': 'Rio Grande do Sul', 
        'RO': 'Rondônia', 'RR': 'Roraima', 'SC': 'Santa Catarina', 'SP': 'São Paulo', 
        'SE': 'Sergipe', 'TO': 'Tocantins'
    }

    while True:
        uf = input("Digite o código do estado (Ex: SP, RJ): ").upper()
        if uf in estados:
            le_uf = LabelEncoder()
            le_uf.fit(['AC', 'AL', 'AP', 'AM', 'BA', 'CE', 'DF', 'ES', 'GO', 'MA', 'MT', 'MS', 'MG', 'PA', 'PB', 'PR', 'PE', 'PI', 'RJ', 'RN', 'RS', 'RO', 'RR', 'SC', 'SP', 'SE', 'TO'])
            uf_valor = le_uf.transform([uf])[0]
            return uf_valor;
        else:
            print("Entrada inválida. Digite um código de estado válido.")

# Função para perguntar o código do município
def ask_municipio():
    while True:
        try:
            municipio = int(input("Digite o código do município (número inteiro): "))
            return municipio
        except ValueError:
            print("Entrada inválida. Digite um número inteiro para o código do município.")

# Função para solicitar as notas do último simulado
def ask_simulado_scores():
    print("Insira as notas do último simulado nas seguintes áreas de conhecimento:")
    
    while True:
        try:
            nota_mt = float(input("Nota em Matemática: "))
            nota_cn = float(input("Nota em Ciências da Natureza: "))
            nota_lc = float(input("Nota em Linguagens: "))
            nota_ch = float(input("Nota em Ciências Humanas: "))
            nota_redacao = float(input("Nota em Redação: "))
            
            return {
                'NU_NOTA_MT': nota_mt,
                'NU_NOTA_CN': nota_cn,
                'NU_NOTA_LC': nota_lc,
                'NU_NOTA_CH': nota_ch,
                'NU_NOTA_REDACAO': nota_redacao
            }
        except ValueError:
            print("Entrada inválida. Por favor, digite valores numéricos para as notas.")

# Função para obter o valor financeiro e seu código
def give_financial_value():
    val = ask_float_input("Renda Familiar (Q006): ")
    code = get_renda_codigo(val)  # Obtém o código correspondente

    le_code = LabelEncoder()
    encoded_code = le_code.fit_transform([code])  

    return encoded_code[0]  # Retorna o primeiro (e único) valor codificado


# Função para prever as notas de um novo aluno
def predict_new_student(knn):
    print("Insira os dados do novo aluno para prever as notas no ENEM:")

    renda_familiar = give_financial_value()
    escolaridade_mae = ask_escolaridade_mae()  # A função já converte a letra
    tipo_escola = ask_int_input("Tipo de Escola (Pública = 0, Privada = 1): ", valid_values=[0, 1])
    cor_raca = ask_etnia()
    uf = ask_uf()  # Perguntar o estado (UF)
    municipio = ask_municipio()  # Perguntar o código do município
     # Solicitar as notas do último simulado
    simulado_scores = ask_simulado_scores()

    # Organizar os dados do aluno em um array numérico
    new_student_data = np.array([
        [
            renda_familiar, 
            escolaridade_mae, 
            tipo_escola, 
            cor_raca, 
            uf, 
            municipio,
            simulado_scores['NU_NOTA_MT'],
            simulado_scores['NU_NOTA_CN'],
            simulado_scores['NU_NOTA_LC'],
            simulado_scores['NU_NOTA_CH'],
            simulado_scores['NU_NOTA_REDACAO']
        ]
    ])
    # Fazer a previsão
    predicted_scores = knn.predict(new_student_data)

    print("Previsão das notas:")
    print(f"Matemática: {predicted_scores[0][0]:.2f}")
    print(f"Ciências da Natureza: {predicted_scores[0][1]:.2f}")
    print(f"Linguagens: {predicted_scores[0][2]:.2f}")
    print(f"Ciências Humanas: {predicted_scores[0][3]:.2f}")
    print(f"Redação: {predicted_scores[0][4]:.2f}")

# Função principal
def main():
    model_path = "modelo_knn.pkl"
    
    # Tentar carregar o modelo treinado
    knn = load_knn_model(model_path)

    # Se o modelo não foi encontrado, treinar e salvar
    if knn is None:
        input_file = "./MICRODADOS_ENEM_2023.csv"  # Arquivo original grande
        output_file = "./MICRODADOS_ENEM_RELEVANTES.csv"  # Novo arquivo com colunas filtradas
        load_principal_columns(input_file, output_file)  # Gerar o novo arquivo
        df = load_and_prepare_data(output_file)  # Carregar os dados relevantes
        df = encode_categorical_columns(df)  # Codificar as colunas
        train_and_save_knn_model(df, model_path)  # Treinar e salvar o modelo
        knn = load_knn_model(model_path)  # Carregar o modelo treinado

    # Fazer previsões para um novo aluno
    predict_new_student(knn)

if __name__ == "__main__":
    main()
