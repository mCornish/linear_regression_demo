import pandas as pd # Python Data Analysis Library (Data structures)
from sklearn import linear_model # Machine Learning tools
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt # Plotting

# Read data
dataframe = pd.read_fwf('brain_body.txt') # Read table data into a DataFrame
x_values = dataframe[['Brain']]
y_values = dataframe[['Body']]

# Train model on data
body_reg = linear_model.LinearRegression() # Create linear regression object
body_reg.fit(x_values, y_values) # Train the model

# Make predictions
body_pred = body_reg.predict(x_values)
print('Mean squared error: %.2f' % mean_squared_error(y_values, body_pred))
print('Variance score: %.2f' % r2_score(y_values, body_pred))

# Visualize results
plt.scatter(x_values, y_values)
plt.plot(x_values, body_reg.predict(x_values))
plt.show()
