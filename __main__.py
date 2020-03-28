from ml_datasets import mnist
from thinc.api import Adam, chain, fix_random_seed, prefer_gpu, Relu, Softmax
from tqdm.notebook import tqdm

if __name__ == "__main__":
    is_gpu = prefer_gpu()
    print("using gpu" if is_gpu else "not using gpu")

    (train_X, train_Y), (dev_X, dev_Y) = mnist()

    print(f"Training size: {len(train_X)}")
    print(f"Dev size: {len(dev_X)}")

    # Neurons per hidden layer
    n_hidden = 32
    # Dropout ratio
    dropout = 0.2

    model = chain(
        Relu(nO=n_hidden, dropout=dropout),
        Relu(nO=n_hidden, dropout=dropout),
        Softmax(),
    )

    [train_X, train_Y, dev_X, dev_Y] = [model.ops.asarray(
        item) for item in [train_X, train_Y, dev_X, dev_Y]]
    
    model.initialize(X=train_X[:5], Y=train_Y[:5])
    nI = model.get_dim("nI")
    nO = model.get_dim("nO")
    print(f"Initialized model with input dimension nI={nI} and output dimension nO={nO}")

    fix_random_seed(0)
    optimizer = Adam(0.001)
    batch_size = 128
    print(f"Measuring performance across iterations:")

    for i in range(10):
        batches = model.ops.multibatch(batch_size, train_X, train_Y, shuffle=True)

        for X, Y in tqdm(batches, leave=False):
            Yh, backpropagate = model.begin_update(X)
            backpropagate(Yh - Y)
            model.finish_update(optimizer)
        
        # Evaluate and print progress
        correct = 0
        total = 0
        
        for X, Y in model.ops.multibatch(batch_size, dev_X, dev_Y):
            Yh = model.predict(X)
            correct += (Yh.argmax(axis=1) == Y.argmax(axis=1)).sum()
            total += Yh.shape[0]
        
        score = correct / total

        print(f" {i} {score:.3f}")

