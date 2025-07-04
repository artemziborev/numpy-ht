import numpy as np
from scipy import sparse
from scipy.special import expit


class LogisticRegression:
    def __init__(self):
        self.w: np.ndarray | None = None
        self.loss_history: list[float] = []

    @staticmethod
    def append_biases(X: np.ndarray | sparse.spmatrix) -> np.ndarray:
        """
        Добавляет столбец единиц для bias.
        """
        if sparse.issparse(X):
            X = X.toarray()
        N = X.shape[0]
        return np.hstack([X, np.ones((N, 1))])

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        return expit(z)

    def _forward(self, X: np.ndarray | sparse.spmatrix) -> np.ndarray:
        if sparse.issparse(X):
            scores = X @ self.w
        else:
            scores = X @ self.w
        return self.sigmoid(scores)

    def loss(self, X: np.ndarray | sparse.spmatrix, y: np.ndarray, reg: float) -> float:
        probs = self._forward(X)
        eps = 1e-15
        probs = np.clip(probs, eps, 1 - eps)
        loss = -np.mean(y * np.log(probs) + (1 - y) * np.log(1 - probs))
        loss += 0.5 * reg * np.sum(self.w ** 2)
        return loss

    def grad(self, X: np.ndarray | sparse.spmatrix, y: np.ndarray, reg: float) -> np.ndarray:
        probs = self._forward(X)
        num_train = X.shape[0]

        if sparse.issparse(X):
            grad = X.T @ (probs - y) / num_train
        else:
            grad = X.T @ (probs - y) / num_train

        grad += reg * self.w
        return grad

    def train(
        self,
        X: np.ndarray | sparse.spmatrix,
        y: np.ndarray,
        learning_rate: float = 1e-3,
        reg: float = 1e-5,
        num_iters: int = 100,
        batch_size: int = 200,
        verbose: bool = False,
    ) -> list[float]:
        num_train, dim = X.shape
        if self.w is None:
            self.w = 0.001 * np.random.randn(dim)

        self.loss_history = []

        for it in range(num_iters):
            batch_indices = np.random.choice(num_train, batch_size, replace=True)
            if sparse.issparse(X):
                X_batch = X[batch_indices].toarray()
            else:
                X_batch = X[batch_indices]

            y_batch = y[batch_indices]

            loss = self.loss(X_batch, y_batch, reg)
            grad = self.grad(X_batch, y_batch, reg)

            self.loss_history.append(loss)
            self.w -= learning_rate * grad

            if verbose and it % 100 == 0:
                print(f"iteration {it}/{num_iters}: loss {loss:.5f}")

        return self.loss_history

    def predict(self, X: np.ndarray | sparse.spmatrix) -> np.ndarray:
        probs = self._forward(X)
        return (probs >= 0.5).astype(int)

    def save_weights(self, path: str) -> None:
        if self.w is not None:
            np.save(path, self.w)
        else:
            raise ValueError("Модель не обучена!")

    def load_weights(self, path: str) -> None:
        self.w = np.load(path)
