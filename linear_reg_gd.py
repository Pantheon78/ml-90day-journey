import numpy as np
import matplotlib.pyplot as plt

class LinearRegressionGD:
    def __init__(self, learning_rate=0.01,n_iterations=100):#this sets how the model will learn before it sees data
    #learning rate:How big each downhill step should be 
    # n_iterations:How many step to take 

        # updating weights and bias to reduce loss
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None # slope of the line 
        self.bias = None # vertical shift more of the y-intercept
        self.loss_history = [] # records how far you are from the valley at each step
    
    def fit(self,x,y):# this is where the model makes guesses, measures how wrong it is and corrects itself
        # initialize weights and bias with no knowledge 
        n_samples,n_features = x.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # gradient descent
        for i in range(self.n_iterations):

            #Prediction = weighted sum of inputs + bias
            y_pred = np.dot(x,self.weights)+ self.bias

            #calculate error
            error = y_pred - y
            #Predicts high positive
            #Predicts low negative

            #calculate gradients
            dw = (2/n_samples) * np.dot(x.T,error)
            db = (2/n_samples) * np.sum(error)

            #update weights 
            self.weights -= self.learning_rate *dw
            self.bias -= self.learning_rate* db

            #Track loss for visualization
            loss = np.mean(error ** 2)
            self.loss_history.append(loss)
            
            #Print progress after every 100 iterations
            if i % 100 == 0:
                print(f"Iteration{i}: Loss={loss:4f}")

    def predict(self,x):
        return np.dot(x, self.weights) + self.bias
        
    def plot_loss(self):

        plt.figure(figsize=(10,6))
        plt.plot(self.loss_history)
        plt.xlabel('Iterations')
        plt.ylabel('Mean Squared Error')
        plt.title('Gradient Descent Convergence')
        plt.grid(True)
        plt.show()
        
            




