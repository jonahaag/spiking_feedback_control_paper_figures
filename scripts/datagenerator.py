import numpy as np

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
