"""Synthetic datasets"""

import numpy as np
import torch
import torchvision
from einops.layers.torch import Rearrange
from src.utils import permutations

from src.dataloaders.base import SequenceDataset


class Copying(SequenceDataset):
    _name_ = "copying"

    @property
    def init_defaults(self):
        return {
            "l_noise": 100,  # number of padding tokens
            "l_memorize": 10,  # number of tokens to memorize
            "n_tokens": 10,  # alphabet size
            "lag": False,
            "variable": False,  # Randomly distribute memorization tokens throughout sequence instead of frontloading them
            "one_hot": False,
            "static": False, # Use a static dataset of size n_train, otherwise always use random data with n_train per epoch
            "n_train": 10000,
            "n_eval": 1000,
        }

    @property
    def d_input(self):
        return self.n_tokens

    @property
    def d_output(self):
        return self.n_tokens

    @property
    def l_output(self):
        return self.l_noise if self.lag else self.l_memorize

    def setup(self):
        from .datasets.copying import CopyingEvalDataset, CopyingTrainDataset

        if self.static: train_cls = CopyingEvalDataset
        else: train_cls = CopyingTrainDataset

        self.dataset_train = train_cls(
            self.l_noise,
            self.l_memorize,
            self.n_tokens,
            samples=self.n_train,
            lag=self.lag,
            variable=self.variable,
            one_hot=self.one_hot,
        )
        self.dataset_val = CopyingEvalDataset(
            self.l_noise,
            self.l_memorize,
            self.n_tokens,
            samples=self.n_eval,
            lag=self.lag,
            variable=self.variable,
            one_hot=self.one_hot,
        )
        self.dataset_test = None


    def __str__(self):
        return f"{self._name_}{self.l_noise}{'v' if self.variable else ''}"


class Adding(SequenceDataset):
    _name_ = "adding"
    d_input = 2
    d_output = 1
    l_output = 0

    @property
    def init_defaults(self):
        return {
            "l_max": 1000,
            "n_samples": 50000,
            "val_split": 0.1,
        }

    def setup(self):
        from .datasets.adding import adding_static_dataset

        self.dataset_train = adding_static_dataset(self.l_max, self.n_samples)
        self.dataset_test = None
        self.split_train_val(self.val_split)

    def __str__(self):
        return f"{self._name_}{self.l_max}"


class Reconstruct(SequenceDataset):
    _name_ = "reconstruct"

    @property
    def init_defaults(self):
        return {
            "l_seq": 1024, # length of total sequence
            "l_mem": 512,  # length to reconstruct
            "dt": 0.001,
            "freq": 1.0,
            "seed": 0,
            "static": False, # Use a static dataset of size n_train, otherwise always use random data with n_train per epoch
            "n_train": 10000,
            "n_eval": 1000,
        }

    @property
    def d_input(self):
        return 1

    @property
    def d_output(self):
        return self.l_mem

    @property
    def l_output(self):
        return 0

    def setup(self):
        from .datasets.reconstruct import ReconstructEvalDataset, ReconstructTrainDataset

        if self.static: train_cls = ReconstructEvalDataset
        else: train_cls = ReconstructTrainDataset

        self.dataset_train = train_cls(
            samples=self.n_train,
            l_seq=self.l_seq,
            l_mem=self.l_mem,
            dt=self.dt,
            freq=self.freq,
            seed=self.seed,
        )
        self.dataset_val = ReconstructEvalDataset(
            samples=self.n_eval,
            l_seq=self.l_seq,
            l_mem=self.l_mem,
            dt=self.dt,
            freq=self.freq,
            seed=self.seed,
        )
        self.dataset_test = None

    def __str__(self):
        raise NotImplementedError


class Delay(SequenceDataset):
    _name_ = "delay"

    @property
    def init_defaults(self):
        return {
            "l_seq": 1024, # length of total sequence
            "n_lag": 1,  # length to reconstruct
            "l_lag": None,  # length to reconstruct
            "dt": 0.001,
            "freq": 100.0,
            "static": False, # Use a static dataset of size n_train, otherwise always use random data with n_train per epoch
            "n_train": 10000,
            "n_eval": 1000,
        }

    @property
    def d_input(self):
        return 1

    @property
    def d_output(self):
        # NOTE: To reproduce numbers from HTTYH paper, set this equal to 4. There was a bug in the implementation at the time
        return self.n_lag

    @property
    def l_output(self):
        return self.l_seq

    def setup(self):
        from .datasets.delay import DelayEvalDataset, DelayTrainDataset

        if self.static: train_cls = DelayEvalDataset
        else: train_cls = DelayTrainDataset

        self.dataset_train = train_cls(
            samples=self.n_train,
            l_seq=self.l_seq,
            n_lag=self.n_lag,
            l_lag=self.l_lag,
            dt=self.dt,
            freq=self.freq,
        )
        self.dataset_val = DelayEvalDataset(
            samples=self.n_eval,
            l_seq=self.l_seq,
            n_lag=self.n_lag,
            l_lag=self.l_lag,
            dt=self.dt,
            freq=self.freq,
        )
        self.dataset_test = None


    def __str__(self):
        return f"{self._name_}{self.l_noise}{'v' if self.variable else ''}"


class Repeat(SequenceDataset):
    _name_ = 'repeat'

    @property
    def init_defaults(self):
        return {
            "n_train": 1000, 
            "n_val": 200,
            "n_test": 200,
            "l_seq": 100,
            "d_seq": 16,
        }

    @property
    def d_input(self):
        return self.d_seq

    @property
    def d_output(self):
        return 2
    
    @property
    def l_output(self):
        return self.l_seq

    def setup(self):
        from .datasets.repeat import RepeatDataset

        self.dataset_train = RepeatDataset(self.n_train, self.l_seq, self.d_seq)
        self.dataset_val = RepeatDataset(self.n_val, self.l_seq, self.d_seq)
        self.dataset_test = RepeatDataset(self.n_test, self.l_seq, self.d_seq)

class RepeatReg(SequenceDataset):
    _name_ = 'repeatreg'

    @property
    def init_defaults(self):
        return {
            "n_train": 1000, 
            "n_val": 200,
            "n_test": 200,
            "l_seq": 100,
            "d_seq": 16,
        }

    @property
    def d_input(self):
        return self.d_seq

    @property
    def d_output(self):
        return 1
    
    @property
    def l_output(self):
        return self.l_seq

    def setup(self):
        from .datasets.repeat import RepeatRegDataset

        self.dataset_train = RepeatRegDataset(self.n_train, self.l_seq, self.d_seq)
        self.dataset_val = RepeatRegDataset(self.n_val, self.l_seq, self.d_seq)
        self.dataset_test = RepeatRegDataset(self.n_test, self.l_seq, self.d_seq)