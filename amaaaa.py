from numpy.lib.stride_tricks import sliding_window_view
from .base import Module
import numpy as np


class Linear(Module):
    def __init__(self, input_dim, output_dim, bias=False, init="uniform"):
        super().__init__(bias)

        self._input_dim = input_dim
        self._output_dim = output_dim
        self.bias = bias

        self.__init_parameters__(init, (input_dim, output_dim), (1, output_dim))
        self.zero_grad()

    def __str__(self):
        return f"Linear({self._input_dim}, {self._output_dim})"

    def zero_grad(self):
        self._gradient["W"] = np.zeros_like(self._parameters["W"])

        if self.bias:
            self._gradient["b"] = np.zeros_like(self._parameters["b"])

    def update_parameters(self, gradient_step=1e-3):
        self._parameters["W"] -= gradient_step * self._gradient["W"]

        if self.bias:
            self._parameters["b"] -= gradient_step * self._gradient["b"]

    def forward(self, X):
        assert X.shape[1] == self._input_dim, ValueError(
            "Les dimensions de X doivent être (batch_size, input_dim)"
        )

        out = np.dot(X, self._parameters["W"])

        if self.bias:
            out += self._parameters["b"]

        return out

    def backward_update_gradient(self, X, delta):
        assert X.shape[1] == self._input_dim, ValueError(
            "Les dimensions de X doivent être (batch_size, input_dim)"
        )

        assert delta.shape == (X.shape[0], self._output_dim), ValueError(
            "Delta doit être de dimension (batch_size, output_dim)"
        )

        self._gradient["W"] += np.dot(X.T, delta)

        if self.bias:
            self._gradient["b"] += np.sum(delta, axis=0)

    def backward_delta(self, X, delta):
        assert delta.shape == (X.shape[0], self._output_dim), ValueError(
            "Delta doit être de dimension (batch_size, output_dim)"
        )

        return np.dot(delta, self._parameters["W"].T)


class Conv1D(Module):
    def __init__(self, k_size, chan_in, chan_out, stride, bias=False, init="uniform"):
        super().__init__(bias)

        self._k_size = k_size
        self._chan_in = chan_in
        self._chan_out = chan_out
        self._stride = stride

        self.__init_parameters__(init, (k_size, chan_in, chan_out), (chan_out))
        self.zero_grad()

    def __str__(self):
        return (
            f"Conv1D({self._k_size}, {self._chan_in}, {self._chan_out}, {self._stride})"
        )

    def zero_grad(self):
        self._gradient["W"] = np.zeros_like(self._parameters["W"])

        if self.bias:
            self._gradient["b"] = np.zeros_like(self._parameters["b"])

    def update_parameters(self, gradient_step=1e-3):
        self._parameters["W"] -= gradient_step * self._gradient["W"]

        if self.bias:
            self._parameters["b"] -= gradient_step * self._gradient["b"]

    def forward(self, X):
        assert X.shape[2] == self._chan_in, ValueError(
            "Les dimensions de X doivent être (batch, lenght, chan_in)"
        )

        batch_size, length, chan_in = X.shape
        dout = (length - self._k_size) // self._stride + 1
        output = np.zeros((batch_size, dout, self._chan_out))

        for i in range(dout):
            window = X[:, i * self._stride : i * self._stride + self._k_size, :]
            output[:, i, :] = np.tensordot(
                window, self._parameters["W"], axes=([1, 2], [0, 1])
            )

        if self.bias:
            output += self._parameters["b"]

        return output

    def backward_update_gradient(self, X, delta):
        batch_size, length, chan_in = X.shape

        assert chan_in == self._chan_in, ValueError(
            "Les dimensions de X doivent être (batch, length, chan_in)"
        )

        dout = (length - self._k_size) // self._stride + 1

        assert delta.shape == (X.shape[0], dout, self._chan_out), ValueError(
            "Delta doit être de dimension (batch, (length-k_size)//stride +1, chan_out)"
        )

        for i in range(dout):
            window = X[:, i * self._stride : i * self._stride + self._k_size, :]
            self._gradient["W"] += np.tensordot(
                delta[:, i, :], window, axes=([0], [0])
            ).transpose((1, 2, 0))

        if self.bias:
            self._gradient["b"] += np.sum(delta, axis=(0, 1))

    def backward_delta(self, X, delta):
        batch_size, length, chan_in = X.shape

        assert chan_in == self._chan_in, ValueError(
            "Les dimensions de X doivent être (batch, lenght, chan_in)"
        )

        dout = (length - self._k_size) // self._stride + 1

        assert delta.shape == (batch_size, dout, self._chan_out), ValueError(
            "Delta doit être de dimension (batch, (length-k_size)/stride +1, chan_out)"
        )

        delta_prev = np.zeros_like(X)

        for i in range(dout):
            delta_i = delta[:, i, :].reshape(batch_size, 1, 1, self._chan_out)

            kernel = self._parameters["W"][::-1, :, :].reshape(
                1, self._k_size, chan_in, self._chan_out
            )

            delta_prev[
                :, i * self._stride : i * self._stride + self._k_size, :
            ] += np.sum(delta_i * kernel, axis=-1)

        return delta_prev


class MaxPool1D(Module):
    def __init__(self, k_size, stride):
        super().__init__(False)

        self._k_size = k_size
        self._stride = stride

    def __str__(self):
        return f"MaxPool1D({self._k_size}, {self._stride})"

    def zero_grad(self):
        pass  # No gradient

    def backward_update_gradient(self, X, delta):
        pass  # No gradient to update

    def update_parameters(self, gradient_step=1e-3):
        pass  # No parameters to update

    def forward(self, X):
        batch_size, length, chan_in = X.shape
        dout = (length - self._k_size) // self._stride + 1

        X_view = np.zeros((batch_size, dout, chan_in, self._k_size))

        for i in range(dout):
            X_view[:, i, :, :] = X[
                :, i * self._stride : i * self._stride + self._k_size, :
            ].transpose((0, 2, 1))

        output = np.max(X_view, axis=-1)
        return output

    def backward_delta(self, X, delta):
        batch_size, length, chan_in = X.shape
        dout = (length - self._k_size) // self._stride + 1

        assert delta.shape == (batch_size, dout, chan_in), ValueError(
            "Delta doit être de dimension (batch, (length-k_size)/stride +1, chan_in)"
        )

        out = np.zeros_like(X)

        for i in range(dout):
            start = i * self._stride
            end = start + self._k_size
            out[:, start:end, :] += delta[:, i : i + 1, :] * (
                X[:, start:end, :] == np.max(X[:, start:end, :], axis=1, keepdims=True)
            )

        return out


class Flatten(Module):
    def __init__(self):
        super().__init__(False)

    def zero_grad(self):
        pass  # No gradient

    def backward_update_gradient(self, X, delta):
        pass  # No gradient to update

    def update_parameters(self, gradient_step=1e-3):
        pass  # No parameters to update

    def forward(self, X):
        return X.reshape(X.shape[0], -1)

    def backward_delta(self, X, delta):
        return delta.reshape(X.shape)
