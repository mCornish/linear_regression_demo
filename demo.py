import pandas as pd # Python Data Analysis Library (Data structures)
from sklearn import linear_model # Machine Learning tools
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt # Plotting

# Read data
body_df = pd.read_fwf('brain_body.txt') # Read table data into a DataFrame
chlng_df = pd.read_csv('challenge_dataset.txt', header=None)

body_x = body_df[['Brain']]
body_y = body_df[['Body']]
chlng_x = chlng_df[[0]]
chlng_y = chlng_df[[1]]

# Train model on data
body_reg = linear_model.LinearRegression() # Create linear regression object
body_reg.fit(body_x, body_y) # Train the model
chlng_reg = linear_model.LinearRegression()
chlng_reg.fit(chlng_x, chlng_y)

# Make predictions
body_pred = body_reg.predict(body_x)
chlng_pred = chlng_reg.predict(chlng_x)
print('Mean squared error: %.2f' % mean_squared_error(chlng_y, chlng_pred))
print('Variance score: %.2f' % r2_score(chlng_y, chlng_pred))

# Visualize results
plt.scatter(chlng_x, chlng_y)
plt.plot(chlng_x, chlng_pred)
plt.show()
