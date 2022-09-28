import numpy as np
import matplotlib.pyplot as plt


# Data
x = np.linspace(0, 1, 101)
y = 1 + x + x * np.random.random(len(x))

# Param prepare
w0_new = 0
w1_new = 0
a = 0.01
MSE = np.array([])

# Gradient descent
for iteration in range(1, 11):
  y_pred = np.array([])
  error = np.array([])
  error_x = np.array([])
  
  w0 = w0_new
  w1 = w1_new
  
  for i in x:
    y_pred = np.append(y_pred, (w0+w1*i))
  
  error = np.append(error, y_pred -y )
  error_x = np.append(error_x, error*x)
  MSE_val = (error**2).mean()
  MSE = np.append(MSE, MSE_val)
  
  # partial gradient w.r.t. w0/w1 respectively
  w0_new = w0-a*np.sum(error)
  w1_new = w1-a*np.sum(error_x)
  

# Output
plt.plot(MSE, 'b-o')
plt.title("MSE per iteration")
plt.xlabel("Iterantions")
plt.ylabel("MSE value")
plt.show()