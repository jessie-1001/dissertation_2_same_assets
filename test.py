import pandas as pd
df = pd.read_csv('garch_copula_forecasts.csv', index_col=0)
print(df[['Gaussian_VaR_99', 'Gaussian_ES_99', 
          'StudentT_VaR_99', 'StudentT_ES_99',
          'Gumbel_VaR_99', 'Gumbel_ES_99',
          'Clayton_VaR_99', 'Clayton_ES_99']].describe())

import matplotlib.pyplot as plt
plt.figure(figsize=(16,5))
plt.plot(df.index, df['Gaussian_VaR_99'], label='VaR (Gaussian)')
plt.plot(df.index, df['Gaussian_ES_99'], label='ES (Gaussian)')
plt.plot(df.index, df['StudentT_VaR_99'], label='VaR (Student-t)')
plt.legend()
plt.title('Rolling VaR/ES Forecasts')
plt.show()

