import math

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def propagate_with_for_loops(w, b, X, Y):

    nx = len(X)       
    m = len(X[0])     
    
    J = 0
    dw = [0] * nx
    db = 0
    
    for i in range(m):
        z = 0
        for j in range(nx):
            z += w[j] * X[j][i]
        z += b
        
        a = sigmoid(z)
        J += -(Y[0][i] * math.log(a) + (1 - Y[0][i]) * math.log(1 - a))
        
        dz = a - Y[0][i]
        for j in range(nx):
            dw[j] += X[j][i] * dz
        db += dz
        
    J = J / m
    for j in range(nx):
        dw[j] = dw[j] / m
    db = db / m
    
    return dw, db, J