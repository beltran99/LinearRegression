# LinearRegression
This is a python implementation of a linear regressor, which models the relationship between a dependent variable (target) and one or more independent variables (features).

A linear regression model makes the assumption of an existing linear relationship between the features and the target.

A linear regressor is defined as:

$$y = b + w_1 x_1 + w_2 x_2 + ... + w_m x_m$$

where $y$ is the target variable, $x_1, x_2, ..., x_m$ the features and $b$ the bias.

To find the optimal values for the regressor coefficients, we use Mean Squared Error (MSE) as the cost function, updating $b$ and $w_i$ through gradient descent so that the mean squared error values tend to an optimal minimal. 

The MSE objective function is defined as the average of squared error that occurred between the predicted $\hat{y}$ values and the actual $y$ values:

$$MSE = \frac{1}{N} \sum_{i}^{N} (y_i - \hat{y}_i) ^ 2$$

