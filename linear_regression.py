import numpy as np

class LinearRegressor:
    """
    A linear regressor models the relationship between a dependent variable (target) and one or more independent variables (features).
    It assumes that there is a linear relationship between the features and the target.
    Use Cases: Prediction, Relationship Analysis, Parameter Estimation
    """
    def __init__(self, lr: float = 1e-3, threshold: float = 1e-5) -> None:
        self.lr = lr
        self.threshold = threshold
        self.old_loss = np.inf
        
    def fit(self, x: np.array, y: np.array, epochs: int = 200):
        if len(x.shape) == 2:
            self.x = x
        else:
            raise ValueError("x must be two-dimensional")
        if len(y.shape) == 1:
            self.y = y
        else:
            raise ValueError("y must be one-dimensional")
        
        self.N = self.x.shape[0]
        self.n_features = self.x.shape[1]
        
        # Xavier/Glorot Initialization
        limit = np.sqrt(6 / (self.n_features + 1))
        self.b = np.random.uniform(-limit, limit)
        self.coefficients = np.random.uniform(-limit, limit, self.n_features)

        self.epochs = epochs

        iterations = 0
        for _ in range(self.epochs):
            self.gradient_descent()
            iterations += 1
            if iterations > 1 and (self.old_loss - self.new_loss) / self.old_loss < self.threshold: # check for convergence by monitoring loss between consecutive epochs
                print(f"Finished at iteration {iterations}")
                break
            self.old_loss = self.new_loss

    def mse(self, y, y_hat):
        return np.sum((y - y_hat) ** 2) / len(list(y))

    def gradient_descent(self):

        coeff_grad = 0
        b_grad = 0
        
        for i in range(self.N):
            xi = self.x[i, :]
            yi = self.y[i]

            yi_hat = self.predict(xi)

            err = yi - yi_hat

            # MSE gradient
            coeff_grad += (-2 / self.N) * (err * xi)
            b_grad += (-2 / self.N) * err
            
        # Parameter update
        self.coefficients = self.coefficients - self.lr * coeff_grad
        self.b = self.b - self.lr * b_grad

        self.new_loss = self.mse(self.y, self.predict(self.x))

    def predict(self, x):
        # normalize data if is not normalized
        mean = np.mean(x, axis=0)
        std = np.std(x, axis=0)
        if not ( np.all(mean.astype(int) == 0) and np.all(std.astype(int) == 1) ):
            x = (x - mean) / std
        return self.b + np.dot(self.coefficients, x.T)


def main():
    from sklearn import datasets
    diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)
    diabetes_X_train = diabetes_X[:-20]
    diabetes_X_test = diabetes_X[-20:]
    diabetes_y_train = diabetes_y[:-20]
    diabetes_y_test = diabetes_y[-20:]
    lin_reg = LinearRegressor(lr=1e-3, threshold=1e-5)
    lin_reg.fit(diabetes_X_train, diabetes_y_train, epochs=10000)
    diabetes_y_pred = lin_reg.predict(diabetes_X_test)


if __name__ == "__main__":
    main()