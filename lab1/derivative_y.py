global_epsilon = 0.000000001

def derivative_y():
    summ = 0
    for i in range(1, 3000):
        x = i * global_epsilon
        y = differentiable_function(x)
        dif = y / x
        summ += dif
    return summ

def differentiable_function(x):
    return x + 1

if __name__ == "__main__":
    print(derivative_y())