import numpy as np
patterns = {
    'sin': np.array([
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 1, 0, 1],
        [0, 1, 0, 1, 1, 1, 1, 1],
        [0, 1, 0, 1, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        ]).flatten(),
    'jim': np.array([
        [0, 0, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0],
        ]).flatten(),
    'gaf': np.array([
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [1, 0, 0, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        ]).flatten()
    # the flatten() function is used to transform the 2D array to a 1D array 
}
labels = {
    'sin': 0,
    'jim': 1,
    'gaf': 2
}


num_classes = len(labels)
num_features = len(next(iter(patterns.values()))) #64
weights = np.zeros((num_classes, num_features))  
alpha = 0.1  
epochs = 100  

# training phase 
for epoch in range(epochs):
    for label, pattern in patterns.items(): #lable represent the letters 'sin, 'jim', and 'gaf' and pattern represent the flattened array of that letter
        target_class = labels[label] 
        for class_index in range(num_classes):
            output = np.sign(np.dot(weights[class_index], pattern)) #perceptron perdiction 
            target_output = 1 if class_index == target_class else -1
            if output != target_output:
                weights[class_index] += alpha * (target_output - output) * pattern

#prints the learned weights after training phase is complete
print("weights:")
for i, w in enumerate(weights, start=1):
    print(f"class {i}: ")
    for j in range(0, 63):
        print(w[j], end=' ')
        # if j % 7 == 0 and j != 0:
        #   print()
    print()