import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from transform import transform
from utils import ObjectView, set_seed

def get_dataset_args(num_samples=5000, seed=42):
    arg_dict = {
        'num_samples': num_samples,
        'train_split': 0.8,
        'template_len': 12,
        'padding': [3, 10],
        'scale_coeff': 0.0, 
        'max_translation': 2,
        'corr_noise_scale': 0.0,
        'iid_noise_scale': 0.02,
        'shear_scale': 0.0,
        'shuffle_seq': True,
        'final_seq_length': 15,
        'seed': seed,
        'url': 'https://github.com/greydanus/mnist1d/raw/master/mnist1d_data.pkl'
    }
    return ObjectView(arg_dict)

def get_templates():
    d0 = np.asarray([5,6,6.5,6.75,7,7,7,7,6.75,6.5,6,5])
    d1 = np.asarray([5,3,3,3.4,3.8,4.2,4.6,5,5.4,5.8,5,5])
    d2 = np.asarray([5,6,6.5,6.5,6,5.25,4.75,4,3.5,3.5,4,5])
    d3 = np.asarray([5,6,6.5,6.5,6,5,5,6,6.5,6.5,6,5])
    d4 = np.asarray([5,4.4,3.8,3.2,2.6,2.6,5,5,5,5,5,5])
    d5 = np.asarray([5,3,3,3,3,5,6,6.5,6.5,6,4.5,5])
    d6 = np.asarray([5,4,3.5,3.25,3,3,3,3,3.25,3.5,4,5])
    d7 = np.asarray([5,7,7,6.6,6.2,5.8,5.4,5,4.6,4.2,5,5])
    d8 = np.asarray([5,4,3.5,3.5,4,5,5,4,3.5,3.5,4,5])
    d9 = np.asarray([5,4,3.5,3.5,4,5,5,5,5,4.7,4.3,5])
    
    x = np.stack([d0,d1,d2,d3,d4,d5,d6,d7,d8,d9])
    x -= x.mean(1, keepdims=True)
    x /= x.std(1, keepdims=True)
    x -= x[:,:1]
    
    templates = {'x': x/6., 't': np.linspace(-5, 5, len(d0))/6.,
                 'y': np.asarray([0,1,2,3,4,5,6,7,8,9])}
    return templates

def generate_mnist1d_raw(args):
    templates = get_templates()
    set_seed(args.seed)
    
    xs, ys = [], []
    samples_per_class = args.num_samples // len(templates['y'])
    
    for label_ix in range(len(templates['y'])):
        for example_ix in range(samples_per_class):
            x = templates['x'][label_ix]
            t = templates['t']
            y = templates['y'][label_ix]
            x, new_t = transform(x, t, args)
            xs.append(x)
            ys.append(y)
    
    batch_shuffle = np.random.permutation(len(ys))
    xs = np.stack(xs)[batch_shuffle]
    ys = np.stack(ys)[batch_shuffle]
    
    # Center and Standardize
    xs = (xs - xs.mean()) / xs.std()
    
    return xs, ys


class SpikingMNIST1D(Dataset):
    """
    A PyTorch dataset for generating binary Poisson-distributed spike data 
    based on the MNIST1D dataset.
    
    Input: Normalized 1D scalar waveforms converted to spike trains.
    Output: Spike trains representing the digit class (0-9).
    """

    def __init__(self, target_fr: np.ndarray, 
                 size: int, seed: int, 
                 T: int = 1000, fmin: int = 10, fmax: int = 100,
                 homogeneous_poisson: bool = False, regular: bool = False,
                 poisson_to_dots: bool = False):
        
        self.rng = set_seed(seed)
        self.class_names = [str(i) for i in range(10)]
        self.size = size
        self.num_classes = 10

        self.T = T
        self.fmin = fmin
        self.fmax = fmax

        self.homogeneous_poisson = homogeneous_poisson
        self.regular = regular
        self.poisson_to_dots = poisson_to_dots

        # Generate the raw scalar data
        # We request slightly more samples to handle rounding in class balancing
        raw_args = get_dataset_args(num_samples=size + 20, seed=seed)
        raw_x, raw_y = generate_mnist1d_raw(raw_args)
        
        # Trim to exact size requested
        raw_x = raw_x[:size]
        raw_y = raw_y[:size]

        self.raw_x = raw_x
        self.x_data = []
        self.y_data = []
        self.labels = []
        self.targets = []
        self.target_fr = []
        
        for i in range(self.size):
            
            # 1. Get raw scalar vector
            scalar_seq = raw_x[i] # Shape: (40,)
            label = raw_y[i]
            
            # 2. Normalize scalar vector to [0, 1] range to map to Firing Rates
            # MNIST1D data is roughly standard normal, so we use min-max scaling per sample
            # to ensure the pattern is preserved but mapped to positive rates.
            seq_min = scalar_seq.min()
            seq_max = scalar_seq.max()
            if seq_max - seq_min > 1e-6:
                norm_seq = (scalar_seq - seq_min) / (seq_max - seq_min)
            else:
                norm_seq = np.zeros_like(scalar_seq)
            
            # 3. Convert normalized scalars into input spike trains
            # val shape is (40,) -> Output shape (T, 40)
            poisson_x = self._get_x_data(norm_seq)
            if self.poisson_to_dots:
                summed_spikes = poisson_x.sum(axis=0) * (1e3 / self.T)
                poisson_x = (summed_spikes - self.fmin) / (self.fmax - self.fmin)
            self.x_data.append(poisson_x)
            
            # 4. Get class label
            self.labels.append(label)
            
            # 5. Get target firing rate vector
            # We assume target_fr is [target_high, target_low]
            # e.g. [10, 0] or [100, 2]
            fr = target_fr[1] * np.ones(self.num_classes)
            fr[label] = target_fr[0]
            self.targets.append(fr)
            
            # 6. Convert target firing rate into spike trains
            self.y_data.append(self._get_y_data(fr))
            
            # 7. Get actual target firing rate
            self.target_fr.append(self._get_target_fr(self.y_data[i][-1]))
    
    def _get_x_data(self, val):
        """
        Converts normalized scalar features into spike trains.
        """
        # val shape: (N_features,) 
        
        # Map [0,1] value to [fmin, fmax] Hz
        freq_range = self.fmax - self.fmin
        
        if self.homogeneous_poisson:
            prob = (self.fmin / 1e3) + (freq_range / 1e3) * val
            # Broadcast prob to (T, N_features)
            poisson_val = (np.random.rand(self.T, len(val)) < prob[np.newaxis, :]).astype(np.float32)
            
        elif self.regular:
            freq = (self.fmin / 1e3) + (freq_range / 1e3) * val
            # Avoid division by zero
            freq = np.clip(freq, 1e-6, None) 
            interv = (1/freq).astype(np.int32)
            
            poisson_val = np.zeros((self.T, len(val)))
            for i in range(len(val)): 
                if interv[i] > 0:
                    poisson_val[:,i][0::interv[i]] = 1.0
            poisson_val = poisson_val.astype(np.float32)
            
        else: # deterministic (inhomogeneous with jitter) Poisson spike trains
            freq = self.fmin + freq_range * val
            num_spikes = np.floor(freq * (self.T / 1000)).astype(int)
            poisson_val = np.zeros((len(freq), self.T), dtype=int)
            
            for i, n in enumerate(num_spikes):
                if n > 0:
                    event_indices = np.linspace(0, self.T, n, endpoint=False).astype(int)
                    # Add jitter
                    jitter = np.random.randint(-2, 3, size=len(event_indices))
                    event_indices = np.clip(event_indices + jitter, 0, self.T - 1)
                    poisson_val[i, event_indices] = 1

            poisson_val = poisson_val.T.astype(np.float32)

        return poisson_val

    def _get_y_data(self, target):
        """
        Generates target spike trains based on class label FR.
        target shape: (10,)
        """
        if self.homogeneous_poisson:
            poisson_val = (np.random.rand(self.T, self.num_classes) < (target / 1e3)).astype(float)
        
        else:
            num_spikes = np.floor(target * (self.T / 1000)).astype(int)
            poisson_val = np.zeros((len(target), self.T), dtype=int)
            for i, n in enumerate(num_spikes):
                if n > 0:
                    event_indices = np.linspace(0, self.T, n, endpoint=False).astype(int)
                    jitter = np.random.randint(-2, 3, size=len(event_indices))
                    event_indices = np.clip(event_indices + jitter, 0, self.T - 1)
                    poisson_val[i, event_indices] = 1

            poisson_val = poisson_val.T.astype(np.float32)

        return poisson_val
    
    def _get_target_fr(self, poisson):
        return poisson.sum(axis=0) * (1e3 / self.T)
    
    def __getitem__(self, index):
        return (
            self.x_data[index].astype(np.float32).copy(),
            self.y_data[index].astype(np.float32).copy(),
            np.float32(self.labels[index]),
            np.float32(self.targets[index]),
            np.float32(self.target_fr[index])
        )

    def __len__(self):
        return len(self.x_data)