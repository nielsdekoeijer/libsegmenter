import torch
import checkCola as cola

class Segmenter(torch.nn.Module):
    def __init__(self, segment_size, hop_size, window, mode="wola", edge_correction=True, normalize_window=True, device=None, dtype=None):
        self.factory_kwargs = {"device": device, "dtype": dtype}
        super(Segmenter, self).__init__()
        self.hop_size = hop_size
        self.segment_size = segment_size
        self.window = window.to(device)

        # asserts to ensure correctness
        if self.segment_size % 2 != 0:
            raise ValueError("only even segment_size is supported")

        if self.hop_size > self.segment_size:
            raise ValueError("hop_size cannot be larger than segment_size")

        if self.window.shape[0] != self.segment_size:
            raise ValueError("specified window must have the same size as segment_size")

        # compute prewindow and postwindow
        self.prewindow = self.window.clone()
        self.postwindow = self.window.clone()
        i = self.hop_size
        for h_idx in range(1, self.segment_size // self.hop_size + 1):
            idx1_start = h_idx * self.hop_size
            idx1_end = self.segment_size
            idx2_start = 0
            idx2_end = self.segment_size - idx1_start
            self.prewindow[idx2_start:idx2_end] = self.prewindow[idx2_start:idx2_end] + self.window[idx1_start:idx1_end]
            self.postwindow[idx1_start:idx1_end] = self.postwindow[idx1_start:idx1_end] + self.window[idx2_start:idx2_end]

        # Perform normalization of window function
        if normalize_window:
            value = cola.checkCola(self.window.numpy(), self.hop_size)
            normalization = value[1]
            self.window = self.window / normalization
            self.prewindow = self.prewindow / normalization
            self.postwindow = self.postwindow / normalization
        
        if edge_correction == False:
            self.prewindow = self.window
            self.postwindow = self.window

        if mode == "wola":
            self.mode = mode
            self.window = torch.sqrt(self.window)
            self.prewindow = torch.sqrt(self.prewindow)
            self.postwindow = torch.sqrt(self.postwindow)
        elif mode == "ola":
            self.mode = mode
        else :
            raise ValueError(f"only support for model ola and wola")



    def segment(self, x):
        if( (x.dim() == 2) & (x.shape[1] > 1)):
            number_of_batch_elements = x.shape[0]
            number_of_samples = x.shape[1]
        elif (((x.dim() == 2) & (x.shape[1] == 1)) | x.dim() == 1):
            number_of_batch_elements = 1
            number_of_samples = x.shape[0]
        else :
            raise ValueError(f"only support for inputs with dimension 1 or 2, provided {x.dim()}")
        
        number_of_segments = (number_of_samples) // self.hop_size - self.segment_size // self.hop_size + 1

        N = x.shape[-1]
        K = (N - self.segment_size) // self.hop_size + 1

        if number_of_batch_elements == 1:
            X = torch.zeros((number_of_segments, self.segment_size), **self.factory_kwargs)
            if self.mode == "wola":
                k = 0
                X[0,:] = self.prewindow * x[k * self.hop_size : k * self.hop_size + self.segment_size]
                for k in range(1,number_of_frames-1):
                    X[k,:] = self.window * x[k * self.hop_size : k * self.hop_size + self.segment_size]
                k = number_of_segments - 1
                X[k,:] = self.postwindow * x[k * self.hop_size : k * self.hop_size + self.segment_size]
            else :
                for k in range(number_of_segments):
                    X = x[x * self.hop_size : k * self.hop_size + self.segment_size]
        else :
            X = torch.zeros((number_of_batch_elements, number_of_segments, self.segment_size), **self.factory_kwargs)
            for b in range(number_of_batch_elements):
                if self.mode == "wola":
                    k = 0
                    X[:,k,:] = x[:,k * self.hop_size : k * self.hop_size + self.segment_size] * self.prewindow
                    for k in range(1,number_of_segments - 1):
                        X[:,k,:] = x[:,k * self.hop_size : k * self.hop_size + self.segment_size] * self.window
                    k = number_of_segments - 1
                    X[:,k,:] = x[:,k * self.hop_size : k * self.hop_size + self.segment_size] * self.postwindow
                else :
                    for k in range(number_of_segments):
                        X[:,k,:] = x[:, k * self.hop_size : k * self.hop_size + self.segment_size]
        return X

    def unsegment(self, X):
        if(X.dim() == 3):
            number_of_batch_elements = X.shape[0]
            number_of_segments = X.shape[1]
        elif (X.dim() == 2):
            number_of_batch_elements = 1
            number_of_segments = X.shape[0]
        else :
            raise ValueError(f"only support for inputs with dimension 2 or 3, provided {X.dim()}")
        number_of_samples = (number_of_segments - 1) * self.hop_size + self.segment_size


        if number_of_batch_elements == 1:
            x = torch.zeros((number_of_samples), **self.factory_kwargs)
            k = 0
            x[k * self.hop_size : k * self.hop_size + self.segment_size] += self.prewindow * X[k]
            for k in range(1, number_of_segments - 1):
                x[k * self.hop_size : k * self.hop_size + self.segment_size] += self.window * X[k]
            k = k + 0
            x[k * self.hop_size : k * self.hop_size + self.segment_size] += self.postwindow * X[k]

        else :
            x = torch.zeros((number_of_batch_elements, number_of_samples), **self.factory_kwargs)
            k = 0
            x[:, k * self.hop_size : k * self.hop_size + self.segment_size] += self.prewindow * X[:, k, :]
            for k in range(1, number_of_segments - 1):
                x[:, k * self.hop_size : k * self.hop_size + self.segment_size] += self.window * X[:, k, :]
            k = k + 1
            x[:, k * self.hop_size : k * self.hop_size + self.segment_size] += self.postwindow * X[:, k, :]
        return x

