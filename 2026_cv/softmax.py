import numpy as np

def softmax_loss_vectorized(W, x, y, reg):
    '''
    loss = 0.0
    dW = np.zeros_like(W)

    num_classes = W.shape[1] 
    num_train = x.shape[0] 

    # 1. Forward Pass
    scores = np.dot(x, W)
    scores -= np.max(scores, axis=1, keepdims=True)
    p = np.exp(scores)
    p /= np.sum(p, axis=1, keepdims=True)

    # 2. Loss 계산
    correct_logprobs = -np.log(p[np.arange(num_train), y])
    loss = np.sum(correct_logprobs) / num_train + reg * np.sum(W * W)

    # 3. Backward Pass (Gradient)
    p[np.arange(num_train), y] -= 1
    dW = np.dot(x.T, p)
    dW = dW / num_train + 2 * reg * W

    return loss, dW
    '''
    loss = 0.0
    dW = np.zeros_like(W)

    num_classes = W.shape[1]
    num_train = x.shape[0]

    # Forward Pass
    scores = np.dot(x, W) # (500, 10)
    # axis=1에 해당하는 column을 축소하면서 진행해라!(column 방향으로 진행하라!) -> 같은 행 중 최댓값을 찾아라!
    scores -= np.max(scores, axis=1, keepdims=True) # (500, 1)
    print(scores.shape)



    return loss, dW