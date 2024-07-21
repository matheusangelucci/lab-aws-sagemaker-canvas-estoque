import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np

# Carregar o dataset
df = pd.read_csv('datasets/dataset-1000-com-preco-variavel-e-renovacao-estoque.csv')

# Converter 'DATA_EVENTO' para datetime
df['DATA_EVENTO'] = pd.to_datetime(df['DATA_EVENTO'])

# Determinar a data mais recente
data_mais_recente = df['DATA_EVENTO'].max()

# Particionar a data para o modelo
df['ANO'] = df['DATA_EVENTO'].dt.year
df['MES'] = df['DATA_EVENTO'].dt.month
df['DIA'] = df['DATA_EVENTO'].dt.day

# Definir as características (features) e o alvo (target)
features = ['ID_PRODUTO', 'PRECO', 'ANO', 'MES', 'DIA']
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

# Gerar novas datas (para os próximos 1000 dias)
novas_datas = pd.date_range(start=data_mais_recente + pd.Timedelta(days=1), periods=1000)

# Selecionar as mesmas características usadas no treinamento
features = ['ID_PRODUTO', 'PRECO', 'ANO', 'MES', 'DIA']
dadosvarpreco = df[features]

# Fazer previsões para o dataset Preço variável
previsoes = model.predict(dadosvarpreco)

# Ajustar previsões para garantir que não sejam negativas
previsoes = np.maximum(previsoes, 0)  # Define o mínimo como 0

# Criar DataFrame de previsões
previsoes_df = pd.DataFrame({
    'ID_PRODUTO': df['ID_PRODUTO'],
    'PRECO': df['PRECO'],
    'DATA_EVENTO': np.tile(novas_datas, 1000 // len(novas_datas) + 1)[:1000],  # Novas datas para cada produto
    'PREVISAO_QUANTIDADE_ESTOQUE': previsoes
})

# Converter a coluna 'PREVISAO_QUANTIDADE_ESTOQUE' para Int
previsoes_df['PREVISAO_QUANTIDADE_ESTOQUE'] = previsoes_df['PREVISAO_QUANTIDADE_ESTOQUE'].astype(int)

# Exportar as previsões para um CSV
previsoes_df.to_csv('previsoes_varpreco.csv', index=False)
