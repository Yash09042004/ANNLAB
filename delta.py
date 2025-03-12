import numpy as np


X = np.array([
    [1, -2, 0, -1], 
    [0, 1.5, -0.5, -1], 
    [-1, 1, 0.5, -1]
])

d = np.array([-1, -1, 1]) 


w = np.array([1, -1, 0, 0.5])  


c = 0.1  

l = 1

def sigmoid(x):
    return 2 / (1 + np.exp(-x)) - 1  


def sigmoid_derivative(y):
    return 0.5 * (1 - y**2) 


for i in range(len(X)):
    net = np.dot(w, X[i]) 
    o = sigmoid(l*net) 
    
    f_prime = sigmoid_derivative(o)  

    w = w + c * (d[i] - o) * f_prime * X[i]

    print(f"Iteration {i+1}:")
    print(f"Net: {net:.2f}, Output: {o:.2f}")
    print(f"Updated Weights: [{', '.join(f'{weight:.2f}' for weight in w)}]")
    print("-" * 50)

print("Final Weights:", [f"{weight:.2f}" for weight in w])



































# def input_array(prompt):
#     print(prompt)
#     rows = int(input("Enter number of rows: "))
#     cols = int(input("Enter number of columns: "))
#     array = []
#     for i in range(rows):
#         row = list(map(float, input(f"Enter row {i+1} values separated by space: ").split()))
#         array.append(row)
#     return np.array(array)

# # Function to take input for the weights
# def input_weights(prompt, size):
#     print(prompt)
#     weights = list(map(float, input(f"Enter {size} weights separated by space: ").split()))
#     return np.array(weights)

# # Taking input from the user
# X = input_array("Enter the input array X:")
# d = np.array(list(map(float, input("Enter the desired output array d separated by space: ").split())))
# w = input_weights("Enter the initial weights w:", X.shape[1])
# c = float(input("Enter the learning rate: "))
