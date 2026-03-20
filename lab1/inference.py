def inference_one_neuron(x_vector, weight_vector, bias):
    summ = 0
    for x, w in zip(x_vector, weight_vector):
        summ += x * w
    return summ + bias