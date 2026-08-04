import numpy as np
from mnist1d.data import generate_mnist1d_raw, get_dataset_args

class DataGenerator:
    """
    A base class for data generation.
    """
    def __init__(self, seed : int, n_samples : int, isValidation : bool):
        np.random.seed(seed)
        # TODO would be cleaner to also use batch size here
        self.n_samples = n_samples
        self.isValidation = isValidation
        self.current_idx = 0
    
    def _create_data(self):
        raise NotImplementedError

    def _get_sample(self):
        raise NotImplementedError
    
    def get_next_sample(self):
        raise NotImplementedError


class BinaryDataGenerator(DataGenerator):
    """
    A class for generating input and target data for spiked-based binary classification.
    """
    def __init__(self, n_samples : int, input_frmin : int, input_frmax : int,  target_frmin : int, target_frmax : int,
                 seed : int, isValidation : bool = False):
        super().__init__(seed, n_samples, isValidation)
        
        self.input_frmin = input_frmin
        self.input_frmax = input_frmax
        self.target_frmin = target_frmin
        self.target_frmax = target_frmax
        self._create_data()

    def _create_data(self):
        """
        Create random input and target data for training.
        """
        self.labels = np.vstack((np.zeros(self.n_samples//2), np.ones(self.n_samples//2))).T.reshape(-1,1) # creates an array [0,1,0,1,..] of length n_samples
        # For validation we want to keep this ordered (one example per class per validation batch)
        if not self.isValidation: 
            np.random.shuffle(self.labels)
        self.inputs = np.hstack(((1 - self.labels) * (self.input_frmax - self.input_frmin) + self.input_frmin,
                                self.labels * (self.input_frmax - self.input_frmin) + self.input_frmin))
        self.targets = np.hstack(((1-self.labels) * (self.target_frmax - self.target_frmin) + self.target_frmin,
                                self.labels * (self.target_frmax - self.target_frmin) + self.target_frmin))
        assert self.inputs.shape == (self.n_samples,2)
        assert self.targets.shape == (self.n_samples,2)

    def _get_sample(self, idx : int):
        """
        Get a single sample of input and target data.
        """
        return self.inputs[[idx],:], self.targets[[idx],:], self.labels[[idx]] # Indexing with an array preserves dimensions
    
    def get_next_sample(self):
        """
        Get the next sample of input and target data and increase the index counter.
        """
        sample = self._get_sample(self.current_idx)
        self.current_idx += 1
        return sample

class YinYangDataGenerator(DataGenerator):
    """
    A class for generating input and target data for spiked-based binary classification.
    Adapted from: https://github.com/lkriener/yin_yang_data_set
    """
    def __init__(self, n_samples : int, input_frmin : int, input_frmax : int, target_frmin : int, target_frmax : int,
                 r_small : float = 0.1, r_big : float = 0.5, seed : int = 42, isValidation : bool = False):
        super().__init__(seed, n_samples, isValidation)

        self.input_frmin = input_frmin
        self.input_frmax = input_frmax
        self.target_frmin = target_frmin
        self.target_frmax = target_frmax
        
        self.r_small = r_small
        self.r_big = r_big
        
        self.labels = np.empty((n_samples, 1))
        self.inputs = np.empty((n_samples, 4))
        self.targets = np.empty((n_samples, 3))
        self.class_names = ['yin', 'yang', 'dot']

        self._create_data()

    def _create_data(self):
        """
        Create random input and target data for training.
        """
        for i in range(self.n_samples):
            # keep num of class instances balanced by using rejection sampling
            # choose class for this sample
            if self.isValidation:
                # We want one example from each class for validation
                goal_class = i%3
            else:
                goal_class = np.random.randint(3)
            x, y, c = self._create_sample(goal=goal_class)
            # add mirrod axis values
            x_flipped = 1. - x
            y_flipped = 1. - y
            # convert coordinates into spike trains
            self.inputs[i, :] = np.array([x, y, x_flipped, y_flipped]) * (self.input_frmax - self.input_frmin) + self.input_frmin
            self.labels[i,0] = c
            self.targets[i, :] = self.target_frmin * np.ones(3) + (self.target_frmax - self.target_frmin) * np.array([i==c for i in range(3)], dtype=int)

    def _get_sample(self, idx : int):
        """
        Get a single sample of input and target data.
        """
        return self.inputs[[idx],:], self.targets[[idx],:], self.labels[[idx]] # Indexing with an array preserves dimensions

    def get_next_sample(self):
        """
        Get the next sample of input and target data and increase the index counter.
        """
        sample = self._get_sample(self.current_idx)
        self.current_idx += 1
        return sample

    def _create_sample(self, goal=None):
        """
        Create a single sample (x,y,c) of the yin-yang data set.
        """
        # sample until goal is satisfied
        found_sample_yet = False
        while not found_sample_yet:
            # sample x,y coordinates
            x, y = np.random.rand(2) * 2. * self.r_big
            # check if within yin-yang circle
            if np.sqrt((x - self.r_big)**2 + (y - self.r_big)**2) > self.r_big:
                continue
            # check if they have the same class as the goal for this sample
            c = self._which_class(x, y)
            if goal is None or c == goal:
                found_sample_yet = True
                break
        return x, y, c

    def _which_class(self, x : float, y : float):
        """
        Returns the class for a sample (x,y).
        """
        # equations inspired by
        # https://link.springer.com/content/pdf/10.1007/11564126_19.pdf
        d_right = self.dist_to_right_dot(x, y)
        d_left = self.dist_to_left_dot(x, y)
        criterion1 = d_right <= self.r_small
        criterion2 = d_left > self.r_small and d_left <= 0.5 * self.r_big
        criterion3 = y > self.r_big and d_right > 0.5 * self.r_big
        is_yin = criterion1 or criterion2 or criterion3
        is_circles = d_right < self.r_small or d_left < self.r_small
        if is_circles:
            return 2
        return int(is_yin)

    def dist_to_right_dot(self, x : float, y : float):
        return np.sqrt((x - 1.5 * self.r_big)**2 + (y - self.r_big)**2)

    def dist_to_left_dot(self, x : float, y : float):
        return np.sqrt((x - 0.5 * self.r_big)**2 + (y - self.r_big)**2)

class Mnist1DDataGenerator(DataGenerator):
    """
    A class for generating input and target data for a spike-based version of MNIST-1D.

    The raw 1D waveforms are produced by the MNIST-1D generator (``mnist1d.py``, ported from
    https://github.com/greydanus/mnist1d): per-class templates passed through the random
    transform pipeline (padding/dilation, translation, correlated + iid noise, shearing) and
    subsampled to ``n_features`` points (15 by default, to match the 15 input neurons).

    For the hardware-in-the-loop setup this generator only produces firing-RATE vectors; the
    on-chip FPGA Poisson generators (together with the trainer's spikegen calibration) turn
    those rates into the actual input/target spikes. Each raw waveform is min-max normalized
    per sample to [0, 1] and scaled to ``[input_frmin, input_frmax]``; the class label is
    one-hot encoded onto ``n_classes`` (10) target neurons in ``[target_frmin, target_frmax]``.

    The public interface (``get_next_sample`` returning ``(inputs, targets, labels)``) matches
    the other data generators in this module so the rest of the pipeline is unchanged.
    """
    def __init__(self, n_samples : int, input_frmin : int, input_frmax : int, target_frmin : int, target_frmax : int,
                 seed : int, isValidation : bool = False, n_features : int = 15, n_classes : int = 10):
        super().__init__(seed, n_samples, isValidation)

        self.seed = seed
        self.input_frmin = input_frmin
        self.input_frmax = input_frmax
        self.target_frmin = target_frmin
        self.target_frmax = target_frmax

        self.n_features = n_features
        self.n_classes = n_classes

        self.inputs = np.empty((n_samples, n_features))
        self.targets = np.empty((n_samples, n_classes))
        self.labels = np.empty((n_samples, 1))
        self.class_names = [str(c) for c in range(n_classes)]

        self._create_data()

    def _create_data(self):
        """
        Generate the raw MNIST-1D waveforms and convert them into input/target firing rates.

        ``generate_mnist1d_raw`` produces class-balanced, seed-shuffled samples (one batch per
        class), so the dataset is balanced regardless of ``isValidation`` (which is kept only
        for interface parity with the other generators); distinct splits come from distinct seeds.
        """
        # Request a class-balanced count (generate_mnist1d_raw uses num_samples // n_classes
        # samples per class), then trim to the exact number requested.
        num_request = int(np.ceil(self.n_samples / self.n_classes) * self.n_classes)
        args = get_dataset_args(num_samples=num_request, seed=self.seed)
        args.final_seq_length = self.n_features

        raw_x, raw_y = generate_mnist1d_raw(args)
        raw_x = raw_x[:self.n_samples]
        raw_y = raw_y[:self.n_samples]

        for i in range(self.n_samples):
            scalar_seq = raw_x[i]
            label = int(raw_y[i])

            # Per-sample min-max normalize to [0, 1], then scale to the input firing-rate range.
            smin, smax = scalar_seq.min(), scalar_seq.max()
            if smax - smin > 1e-6:
                norm_seq = (scalar_seq - smin) / (smax - smin)
            else:
                norm_seq = np.zeros_like(scalar_seq)
            self.inputs[i, :] = norm_seq * (self.input_frmax - self.input_frmin) + self.input_frmin

            # One-hot target firing rates.
            self.labels[i, 0] = label
            self.targets[i, :] = self.target_frmin + (self.target_frmax - self.target_frmin) \
                * np.array([j == label for j in range(self.n_classes)], dtype=int)

    def _get_sample(self, idx : int):
        """
        Get a single sample of input and target data.
        """
        return self.inputs[[idx],:], self.targets[[idx],:], self.labels[[idx]] # Indexing with an array preserves dimensions

    def get_next_sample(self):
        """
        Get the next sample of input and target data and increase the index counter.
        """
        sample = self._get_sample(self.current_idx)
        self.current_idx += 1
        return sample