import numpy as np

class LinearRegressionNormalEq:
    def __init__(self):
        self.weights = None

    def fit(self,x,y):
            # add bias term
            # X.shape[0] tells us how many rows (data points) we have.
            # np.ones((..., 1)) creates a column of 1s of that same height
            # np.c_ concatenates (glues) that column of 1s to the front of your data $X$.
            x_b= np.c_[np.ones((x.shape[0],1)),x]
            # normal equation: θ  = (X^T X)^-1 X^T y
            self.weights = np.linalg.inv(x_b.T.dot(x_b)).dot(x_b.T).dot(y)

    def predict(self,x):
        x_b = np.c_[np.ones((x.shape[0],1)), x]
        return x_b.dot(self.weights)


if __name__ == "__main__":
    print("=" *60)
    print("Test 1: Perfect Linear Relatioship (y=2x)")
    print("=" * 60)

    #Create perfect linear data: y=2x 
    x_test = np.array([[1],[2],[3]])
    y_test = np.array([2,4,6])

    print(f"Training data shape:x{x_test.shape}, y{y_test.shape}")
    print(f"x values: {x_test.flatten()}")
    print(f"y values: {y_test}")


    #create model instance
    model = LinearRegressionNormalEq()
    print("\n1. Model created succesfully")

    #fit model to data
    model.fit(x_test,y_test)
    print(f"\n2. Model trained.Weights: {model.weights}")
    print(f" Expected weights: [0.0,2.0] (for y =0 + 2*x)")

    #make predictions
    predictions  = model.predict(x_test)
    print(f"\n3.Predicctions : {predictions}")

    #calculate and display error 
    mse = np.mean((predictions - y_test)**2)
    print(f"n4. Mean Squared Error:{mse:2e}")

    #validation
    if np.allclose(model.weights, [0,2],atol =1e-10):
        print("\n Test 1 passed: Weights are correct!")
    else:
        print("\n Test 1 failed:Weights are incorrect.")
    print("\n" + "=" * 60)
    print("Test 2: Noisy Linear Relationship")
    print("=" * 60)

    #Create noisy Linear data: y = 4 +3x +noise

    np.random.seed(42) #for reproducibility

    x_noisy = 2 * np.random.rand(10,1)
    y_noisy = 4 + 3 * x_noisy + np.random.randn(10,1) * 0.5 # add noise

    #flatten y_noisy to 1D array 
    y_noisy = y_noisy.flatten()

    model2 = LinearRegressionNormalEq()
    model2.fit(x_noisy,y_noisy)

    print("\n Sample predictions VS actual (first 3 points):")
    for i in range (3):
        pred = model2.predict (x_noisy[i:i+1])
        print(f" x={x_noisy[i][0]:.2f}, pred={pred[0]:.2f}, Actual={y_noisy[i]:.2f}")
    

