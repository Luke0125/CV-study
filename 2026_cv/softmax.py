import numpy as np

def softmax_loss_vectorized(W, x, y, reg):
    '''
    loss = 0.0
    dW = np.zeros_like(W)

    num_classes = W.shape[0]
    num_train = x.shape[0]
    print(num_classes.shape)

    return loss, dW
    '''
    def softmax_loss_vectorized(W, x, y, reg):
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

    # [핵심!] 이 부분이 반드시 있어야 합니다.
    return loss, dW