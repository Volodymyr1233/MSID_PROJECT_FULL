import numpy as np


class LogisticRegrGradient:
    def __init__(self, lr=0.01, epochs=100, batch_size=64, l1_lam=0, l2_lam=0):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.wages = None
        self.l1_lam = l1_lam
        self.l2_lam = l2_lam
        self.train_loss = []
        self.val_loss = []

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def fit(self, x, y, x_test=None, y_test=None):
        m, n = x.shape
        x_bias = np.hstack([np.ones((m, 1)), x])
        self.wages = np.zeros(x_bias.shape[1])

        for epoch in range(self.epochs):
            indices = np.random.permutation(m)
            x_shuffled = x_bias[indices]
            y_shuffled = y[indices]

            for start in range(0, m, self.batch_size):
                end = start + self.batch_size
                xb = x_shuffled[start:end]
                yb = y_shuffled[start:end]

                preds = self.sigmoid(xb @ self.wages)
                error = preds - yb
                gradient = xb.T @ error / len(yb)

                #REGULARYZACJA L1 (bez wyrazu wolnego)
                gradient[1:] += self.l1_lam * np.sign(self.wages[1:])

                #REGULARYZACJA L2 (bez wyrazu wolnego)
                gradient[1:] += self.l2_lam * self.wages[1:]

                self.wages -= self.lr * gradient

            train_preds = self.sigmoid(x_bias @ self.wages)
            self.train_loss.append(self.compute_cross_entropy(y, train_preds))

            if x_test is not None and y_test is not None:
                x_test_bias = np.hstack([np.ones((x_test.shape[0], 1)), x_test])
                test_preds = self.sigmoid(x_test_bias @ self.wages)
                self.val_loss.append(self.compute_cross_entropy(y_test, test_preds))

    def predict(self, x):
        x_bias = np.hstack([np.ones((x.shape[0], 1)), x])
        probs = self.sigmoid(x_bias @ self.wages)
        return (probs >= 0.5).astype(int), probs

    def compute_cross_entropy(self, y_true, y_pred):
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))