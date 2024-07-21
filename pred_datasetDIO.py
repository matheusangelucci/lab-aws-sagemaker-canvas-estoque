import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np

# Carregar o dataset
df = pd.read_csv('datasets\dataset-500-curso-sagemaker-canvas-dio.csv')

# Renomear a coluna 'DIA' para 'DATA'
df = df.rename(columns={'DIA': 'DATA'})

# Converter 'DATA' para datetime
df['DATA'] = pd.to_datetime(df['DATA'])

# Determinar a data mais recente
data_mais_recente = df['DATA'].max()

# Particionar a data para o modelo
df['ANO'] = df['DATA'].dt.year
df['MES'] = df['DATA'].dt.month
df['DIA'] = df['DATA'].dt.day

# Definir as características (features) e o alvo (target)
features = ['ID_PRODUTO', 'ANO', 'MES', 'DIA']
target = 'QUANTIDADE_ESTOQUE'

X = df[features]
y = df[target]

# Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar o modelo
model = LinearRegression()
model.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = model.predict(X_test)

# Calcular métricas
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

#print(f'Mean Absolute Error: {mae}')
#print(f'R² Score: {r2}')

feature_importances = pd.DataFrame(model.coef_, index=features, columns=['Importance'])
#print(feature_importances)

# -----------------------------------------------------------------------------------------------

# Gerar novas datas (para os próximos 500 dias)
novas_datas = pd.date_range(start=data_mais_recente + pd.Timedelta(days=1), periods=500)

# Selecionar as mesmas características usadas no treinamento
features = ['ID_PRODUTO', 'ANO', 'MES', 'DIA']
dadosDIO = df[features]

# Fazer previsões para o dataset DIO
previsoes = model.predict(dadosDIO)

# Ajustar previsões para garantir que não sejam negativas
previsoes = np.maximum(previsoes, 0)  # Define o mínimo como 0

# Criar DataFrame de previsões
previsoes_df = pd.DataFrame({
    'ID_PRODUTO': df['ID_PRODUTO'],
    'DATA': np.tile(novas_datas, 500 // len(novas_datas) + 1)[:500],  # Repetir novas datas para cada produto
    'PREVISAO_QUANTIDADE_ESTOQUE': previsoes
})

# Converter a coluna 'PREVISAO_QUANTIDADE_ESTOQUE' para Int
previsoes_df['PREVISAO_QUANTIDADE_ESTOQUE'] = previsoes_df['PREVISAO_QUANTIDADE_ESTOQUE'].astype(int)

# Exportar as previsões para um CSV
previsoes_df.to_csv('previsoes_DIO.csv', index=False)