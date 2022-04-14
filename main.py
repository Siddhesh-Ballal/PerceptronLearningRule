#Perceptron Learning Rule
import numpy as np
  
def Phi(v):
    if v > 0:
        return 1
    else:
        return -1
    
def perceptronModel(x, w, b):
    v = np.dot(w, x) + b
    print("v = ",v)
    y = Phi(v)
    print("y = phi(v) = ", y)
    return y

alpha = 1
test = {}
test[0] = np.array([1, 1])
test[1] = np.array([1, -1])
test[2] = np.array([-1, 1])
test[3] = np.array([-1, -1])
yt = [1, -1, -1, -1]
print(test)
w = np.array([0, 0])
b = 0

i=0
while(True):
    y = perceptronModel(test[i], w, b)
    if y == yt[i]:
        break
    w = np.add(w, (alpha*test[i]*yt[i]))
    b = b + (alpha*yt[i])
    i+=1           

print("No need to update weights further. Final weights and bias are:")
print("w = ", w)
print("b = ", b)