import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

dataset1 = pd.read_csv('D:/JJ/Oasis Infobyte/2_Unemployment Analysis with Python/archive (3)/Unemployment in India.csv')

label_encoder = LabelEncoder()
dataset1['Frequency'] = label_encoder.fit_transform(dataset1['Frequency'])
dataset1['Area'] = label_encoder.fit_transform(dataset1['Area'])

X = dataset1[['Estimated Employed', 'Estimated Labour Participation Rate (%)', 'Frequency', 'Area']]
y = dataset1['Estimated Unemployment Rate (%)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')


plt.scatter(y_test, y_pred, color='blue')
plt.xlabel('Actual Unemployment Rate (%)')
plt.ylabel('Predicted Unemployment Rate (%)')
plt.title('Unemployment Rate Prediction')
plt.show()

plt.scatter(X_test['Estimated Employed'], y_test, color='blue')
plt.scatter(X_test['Estimated Employed'], y_pred, color='red')
plt.xlabel('Estimated Employed')
plt.ylabel('Unemployment Rate')
plt.title('Unemployment Rate vs. Estimated Employed')
plt.show()


plt.scatter(X_test['Estimated Labour Participation Rate (%)'], y_test, color='blue')
plt.scatter(X_test['Estimated Labour Participation Rate (%)'], y_pred, color='red')
plt.xlabel('Estimated Labour Participation Rate (%)')
plt.ylabel('Unemployment Rate')
plt.title('Unemployment Rate vs. Labour Participation Rate')
plt.show()
