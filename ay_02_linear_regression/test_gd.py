import numpy as np
from linear_reg_gd import LinearRegressionGD

#generate synthetic data
np.random.rand(100,1) # 100 samples,1 feature
x = 2 * np.random.rand(100,1)
y = 5 + 3 * x + np.random.randn(100,1) * 0.5 # true relationshipe +noise
#Bias = 5
#Weight =3
#0.5 = noise 

print("Data shpe:", x.shape,y.shape)
print("First 5 x values:", x[:5].ravel())
print("First 5 y values:", y[:5].ravel())

# create and train model
print("\n --- Training  Gradient Descent Model ---")
model = LinearRegressionGD(learning_rate=0.01,n_iterations = 500)
model.fit(x , y.ravel()) #y.ravel converts (100,1) to (100,) just makes printing clean

print(f"\n ---Results---")
print(f"True relationship: 5 + 3*x + noise")# remainder of the model i created 
print(f"Learned weights:{model.weights}")
print(f"Learned bias:{model.bias}")
print(f"Final loss:{model.loss_history[-1]:.4f}")#loss at the bottom of the hill 

model.plot_loss()
print("\n---predictions---")
x_new = np.array([[0.5],[1.5],[2.5]])
predictions = model.predict(x_new)#uses learned weights and bias 
for x_val,pred in zip(x_new.ravel(),predictions):#loops through the values and pairs each x wih its prediction
    true_y = 5 + 3 * x_val  #true formula no noise
    print(f"x = [x_val:.1f]: Predicted={pred:.2f},True={true_y:2f},Error={abs(pred-true_y):.2f}")