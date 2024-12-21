import numpy as np
import scipy.interpolate as interpolate
import pdb

class DatasetNormalizer:

    def __init__(self, dataset, normalizer):
        dataset = flatten(dataset)
        self.observation_dim = dataset["observations"].shape[1]
        if type(normalizer) == str:
            normalizer = eval(normalizer)

        self.normalizers = {}
        for key, val in dataset.items():
            try:
                self.normalizers[key] = normalizer(val)  # 等于cdfnormalizer  
            except:
                print(f'[ utils/normalization ] Skipping {key} | {normalizer}')
            # key: normalizer(val)
            # for key, val in dataset.items()

    def __repr__(self):
        string = ''
        for key, normalizer in self.normalizers.items():
            string += f'{key}: {normalizer}]\n'
        return string

    def __call__(self, *args, **kwargs):
        return self.normalize(*args, **kwargs)

    def normalize(self, x, key):
        return self.normalizers[key].normalize(x)
# CDF normalizer (dataset.obs)
    def unnormalize(self, x, key):
        return self.normalizers[key].unnormalize(x)

def flatten(dataset):
    '''
        flattens dataset of { key: [ n_episodes x max_path_lenth x dim ] }
            to { key : [ (n_episodes * sum(path_lengths)) x dim ]}
    '''
    flattened = {}
    agent_total_obses = []
    num_oppo_policy = len(dataset)
    for i in range(num_oppo_policy):
        for traj in dataset[i]:
            if traj[0]['observations'].shape[0] < 100:
                s = np.concatenate([traj[0]['observations'],traj[0]['next_observations'][-1].reshape(1,-1)],axis = 0)
            else :
                s = traj[0]['observations']
            agent_total_obses.append(s)
    agent_total_obses = np.concatenate(agent_total_obses, axis=0)
    flattened["observations"] = agent_total_obses       
    return flattened


class Normalizer:
    '''
        parent class, subclass by defining the `normalize` and `unnormalize` methods
    '''

    def __init__(self, X):
        self.X = X.astype(np.float32)
        self.mins = X.min(axis=0)
        self.maxs = X.max(axis=0)

    def __repr__(self):
        return (
            f'''[ Normalizer ] dim: {self.mins.size}\n    -: '''
            f'''{np.round(self.mins, 2)}\n    +: {np.round(self.maxs, 2)}\n'''
        )

    def __call__(self, x):
        return self.normalize(x)

    def normalize(self, *args, **kwargs):
        raise NotImplementedError()

    def unnormalize(self, *args, **kwargs):
        raise NotImplementedError()


class CDFNormalizer(Normalizer):
    '''
        makes training data uniform (over each dimension) by transforming it with marginal CDFs
    '''

    def __init__(self, X):
        super().__init__(atleast_2d(X))
        self.dim = self.X.shape[1]
        self.cdfs = [CDFNormalizer1d(self.X[:, i]) for i in range(self.dim)]

    def __repr__(self):
        return f'[ CDFNormalizer ] dim: {self.mins.size}\n' + '    |    '.join(
            f'{i:3d}: {cdf}' for i, cdf in enumerate(self.cdfs)
        )

    def wrap(self, fn_name, x):
        shape = x.shape
        ## reshape to 2d
        x = x.reshape(-1, self.dim)
        out = np.zeros_like(x)
        for i, cdf in enumerate(self.cdfs):
            fn = getattr(cdf, fn_name)
            out[:, i] = fn(x[:, i])
        return out.reshape(shape)

    def normalize(self, x):
        return self.wrap('normalize', x)

    def unnormalize(self, x):
        return self.wrap('unnormalize', x)

class CDFNormalizer1d:
    '''
        CDF normalizer for a single dimension
    '''

    def __init__(self, X):
        assert X.ndim == 1
        self.X = X.astype(np.float32)
        if self.X.max() == self.X.min():
            self.constant = True
        else:
            self.constant = False
            quantiles, cumprob = empirical_cdf(self.X)
            self.fn = interpolate.interp1d(quantiles, cumprob)
            self.inv = interpolate.interp1d(cumprob, quantiles)

            self.xmin, self.xmax = quantiles.min(), quantiles.max()
            self.ymin, self.ymax = cumprob.min(), cumprob.max()

    def __repr__(self):
        return (
            f'[{np.round(self.xmin, 2):.4f}, {np.round(self.xmax, 2):.4f}'
        )

    def normalize(self, x):
        if self.constant:
            return x

        x = np.clip(x, self.xmin, self.xmax)
        ## [ 0, 1 ]
        y = self.fn(x)
        ## [ -1, 1 ]
        y = 2 * y - 1
        return y

    def unnormalize(self, x, eps=1e-4):
        '''
            X : [ -1, 1 ]
        '''
        ## [ -1, 1 ] --> [ 0, 1 ]
        if self.constant:
            return x

        x = (x + 1) / 2.

        if (x < self.ymin - eps).any() or (x > self.ymax + eps).any():
            print(
                f'''[ dataset/normalization ] Warning: out of range in unnormalize: '''
                f'''[{x.min()}, {x.max()}] | '''
                f'''x : [{self.xmin}, {self.xmax}] | '''
                f'''y: [{self.ymin}, {self.ymax}]'''
            )

        x = np.clip(x, self.ymin, self.ymax)

        y = self.inv(x)
        return y

def empirical_cdf(sample):
    ## https://stackoverflow.com/a/33346366

    # find the unique values and their corresponding counts
    quantiles, counts = np.unique(sample, return_counts=True)

    # take the cumulative sum of the counts and divide by the sample size to
    # get the cumulative probabilities between 0 and 1
    cumprob = np.cumsum(counts).astype(np.double) / sample.size

    return quantiles, cumprob

def atleast_2d(x):
    if x.ndim < 2:
        x = x[:,None]
    return x
