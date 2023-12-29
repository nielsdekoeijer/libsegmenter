import torch

class Segmenter(torch.nn.Module):
    def __init__(self, segment_size, hop_size, window, perform_validation=True, validation_ser=50.0, normalize_window=True):
        super(Segmenter, self).__init__()
        self.hop_size = hop_size
        self.segment_size = segment_size
        self.window = window

        # asserts to ensure correctness
        if self.segment_size % 2 != 0:
            raise ValueError("only even segment_size is supported")

        if self.hop_size > self.segment_size:
            raise ValueError("hop_size cannot be larger than segment_size")

        if self.window.shape[0] != self.segment_size:
            raise ValueError("specified window must have the same size as segment_size")

        # compute prewindow and postwindow
        self.prewindow = self.window.clone()
        i = self.hop_size
        while i < self.segment_size:
            self.prewindow[:-i] += self.window[i:]
            i += self.hop_size

        self.postwindow = self.window.clone()
        i = self.hop_size
        while i < self.segment_size:
            self.postwindow[i:] += self.window[:-i]
            i += self.hop_size

        # this is a tiny bit hacked, but it works well in practise
        if normalize_window:
            normalization = self.prewindow[0]
            self.window = self.window / normalization
            self.prewindow = self.prewindow / normalization
            self.postwindow = self.postwindow / normalization

        # here we ensure that the forward and backward pass have perfrect reconstruction
        if perform_validation:
            # here we assert that the window doesn't have a "DC" component
            # if we are to change the frames individually, such a DC component breaks reconsturction
            # (imagine problems that exist with e.g. a boxcar window, they will exist for hamming too!)
            if abs(self.window[0]) > 1e-8 or abs(self.window[-1]) > 1e-8:
                raise ValueError("filter non-zero at first and / or last entry")

            # next we assert that the reconstruction forward backward is below some ser in db
            # not batched
            itest = torch.randn(segment_size + 10 * hop_size)
            otest = self.unsegment(self.segment(itest))
            ser = 20 * torch.log10(torch.norm(otest)) - 20 * torch.log10(torch.norm(itest - otest))
            if ser < validation_ser:
                raise ValueError(f"segmentation results in {ser} dB signal to error ratio when max allowable ratio is {validation_ser} dB")

            # batched
            itest = torch.randn((2, segment_size + 10 * hop_size))
            otest = self.unsegment(self.segment(itest))
            ser = 20 * torch.log10(torch.norm(otest)) - 20 * torch.log10(torch.norm(itest - otest))
            if ser < validation_ser:
                raise ValueError(f"segmentation results in {ser} dB signal to error ratio when max allowable ratio is {validation_ser} dB")

    def segment(self, x):
        N = x.shape[-1]
        K = (N - self.segment_size) // self.hop_size + 1

        if len(x.shape) == 1:
            X = torch.zeros((K, self.segment_size))
            for k in range(K):
                X[k,:] = x[k * self.hop_size : k * self.hop_size + self.segment_size]
            return X

        if len(x.shape) == 2:
            B = x.shape[0]
            X = torch.zeros((B, K, self.segment_size))
            for k in range(K):
                X[:,k,:] = x[:, k * self.hop_size : k * self.hop_size + self.segment_size]
            return X

        raise ValueError(f"only support for inputs with dimension 1 or 2, provided {len(x.shape)}")

    def unsegment(self, X):
        K = X.shape[-2]
        N = (K - 1) * self.hop_size + self.segment_size

        if len(X.shape) == 2:
            x = torch.zeros((N))
            k = 0
            x[k * self.hop_size : k * self.hop_size + self.segment_size] += self.prewindow * X[k]
            for k in range(1, K - 1):
                x[k * self.hop_size : k * self.hop_size + self.segment_size] += self.window * X[k]
            k = k + 1
            x[k * self.hop_size : k * self.hop_size + self.segment_size] += self.postwindow * X[k]
            return x

        if len(X.shape) == 3:
            B = X.shape[0]
            x = torch.zeros((B, N))
            k = 0
            x[:, k * self.hop_size : k * self.hop_size + self.segment_size] += self.prewindow * X[:, k, :]
            for k in range(1, K - 1):
                x[:, k * self.hop_size : k * self.hop_size + self.segment_size] += self.window * X[:, k, :]
            k = k + 1
            x[:, k * self.hop_size : k * self.hop_size + self.segment_size] += self.postwindow * X[:, k, :]
            return x

        raise ValueError(f"only support for inputs with dimension 2 or 3, provided {len(x.shape)}")

if __name__ == "__main__":
    # quick unit tests
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # 1: valid window should create OK
    try: 
        window = torch.hann_window(1024, periodic=False)
        segmenter = Segmenter(1024, 512, window)
        assert True
        print(f"passed test 1")
    except Exception as e:
        print(e)
        print(f"failed test 1")

    # 2: more difficult window
    try: 
        window = torch.hann_window(3000, periodic=False)
        segmenter = Segmenter(3000, 1000, window)
        assert True
        print(f"passed test 2")
    except Exception as e:
        print(e)
        print(f"failed test 2")

    # 3: messing up periodicity
    try: 
        window = torch.hann_window(1024, periodic=True)
        segmenter = Segmenter(1024, 512, window)
        assert True
        print(f"failed test 3")
    except Exception as e:
        print(f"passed test 3")
