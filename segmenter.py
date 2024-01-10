# single-file implementation of a segmentation class
# can perform OLA and WOLA depending on the input configuration

import torch

def _segmenter_validate_tensor_window(window, segment_size):
    if window.shape != (segment_size):
        raise ValueError(f"provided iwindow tensor shape is {iwindow.shape}, expected {(segment_size)}")

def _segmenter_lookup_window(string):
    None

def _segmenter_compute_begwindow(segment_size, hop_size, window):
    begwindow = window.clone()
    i = hop_size
    while i < segment_size:
        begwindow[:-i] += window[i:]
        i += hop_size

def _segmenter_compute_endwindow(segment_size, hop_size, window):
    endwindow = window.clone()
    i = hop_size
    while i < segment_size:
        endwindow[i:] += window[:-i]
        i += hop_size

class Segmenter(torch.nn.Module):
    """
    docstring
    """
    def __init__(
            self, 
            segment_size: int, 
            hop_size: int, 
            iwindow: torch.tensor, 
            owindow: torch.tensor, 
            auto_normalize_windows: bool = True,
            apply_boundary_corrections: bool = True,
            check_perfect_reconstruction: bool = True, 
            check_perfect_reconstruction_ser_db: float = 50.0, 
            check_cola: bool = True, 
            device=None,
            dtype=None,
        ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(Segmenter, self).__init__()

        # validate inputs
        self.segment_size = segment_size
        if self.segment_size % 2 != 0:
            raise ValueError(f"only even segment_size is supported")

        # hop sizes can never exceed segment size
        self.hop_size = hop_size
        if self.hop_size > self.segment_size:
            raise ValueError(f"hop_size cannot be larger than segment_size")
        
        # we allow the user to use strings to use a predefined window (recommended)
        if isinstance(iwindow, str):
            iwindow = _segmenter_lookup_window(iwindow)

        if isinstance(owindow, str):
            owindow = _segmenter_lookup_window(owindow)

        # we validate window sizes
        if isinstance(iwindow, torch.Tensor):
            _segmenter_validate_tensor_window(iwindow, segment_size)
        else:
            raise ValueError(f"invalid iwindow, expected either a str or a torch.Tensor")

        if isinstance(owindow, torch.Tensor):
            _segmenter_validate_tensor_window(owindow, segment_size)
        else:
            raise ValueError(f"invalid owindow, expected either a str or a torch.Tensor")

        # compute beg(in)window and endwindow
        self.apply_boundary_corrections = apply_boundary_corrections
        if self.apply_boundary_corrections:
            self.ibegwindow = _segmenter_compute_begwindow(self.segment_size, self.hop_size, self.iwindow)
            self.iendwindow = _segmenter_compute_endwindow(self.segment_size, self.hop_size, self.iwindow)
            self.obegwindow = _segmenter_compute_begwindow(self.segment_size, self.hop_size, self.owindow)
            self.oendwindow = _segmenter_compute_endwindow(self.segment_size, self.hop_size, self.owindow)
        else:
            self.ibegwindow = self.iwindow.clone()
            self.iendwindow = self.iwindow.clone()
            self.obegwindow = self.owindow.clone()
            self.oendwindow = self.owindow.clone()

        # this is a tiny bit hacked, but it works well in practise
        self.auto_normalize_windows = auto_normalize_windows
        self.normalization_factor = None
        if self.auto_normalize_windows:
            self.normalization_factor = self.prewindow[0]
            self.window = self.window / self.normalization_factor 
            self.prewindow = self.prewindow / self.normalization_factor
            self.postwindow = self.postwindow / self.normalization_factor

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
