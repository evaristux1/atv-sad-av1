import pandas as pd

arquivo_csv = '../MICRODADOS_ENEM_2023.csv' 

# Carrega os dados do arquivo CSV, especificando o separador como ';'
dados = pd.read_csv(arquivo_csv, encoding='latin1', sep=';')

# Exibe as colunas disponíveis
# print("Colunas disponíveis no arquivo CSV:")
# print(dados.columns)

# Colunas de interesse
colunas_notas = [
    'NU_NOTA_MT',     # Nota em Matemática e suas Tecnologias
    'NU_NOTA_CN',     # Nota em Ciências da Natureza e suas Tecnologias
    'NU_NOTA_LC',     # Nota em Linguagens, Códigos e suas Tecnologias
    'NU_NOTA_CH',     # Nota em Ciências Humanas e suas Tecnologias
    'NU_NOTA_REDACAO' # Nota da Redação (ajustado para o nome correto)
]

colunas_socioeconomicas = [
    'Q006',           # Renda Familiar
    'Q002',           # Escolaridade da Mãe
    'TP_ESCOLA',      # Tipo de Escola (Pública ou Privada)
    'TP_COR_RACA'     # Raça/Cor
]

colunas_localidade = [
    'SG_UF_PROVA',           # Estado
    'CO_MUNICIPIO_PROVA'     # Município
]

# Exibe as colunas agrupadas
print("\nNotas nas Áreas do Conhecimento: \n")
print(dados[colunas_notas])  # Mostra as primeiras linhas das notas

print("Informações Socioeconômicas: \n")
print(dados[colunas_socioeconomicas])  # Mostra as primeiras linhas das informações socioeconômicas

print("Localidade: \n")
print(dados[colunas_localidade])  # Mostra as primeiras linhas da localidade

# Juntando todas as colunas de interesse em um novo DataFrame
dados_combinados = dados[colunas_notas + colunas_socioeconomicas + colunas_localidade]

# Exportando para um novo arquivo CSV
dados_combinados.to_csv('dados_combinados.csv', index=False, encoding='latin1')