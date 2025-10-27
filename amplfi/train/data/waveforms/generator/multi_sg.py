import torch
from .sg import SGGenerator

class MultiSGGenerator(SGGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, **parameters):
        cross, plus = self.sine_gaussian(**parameters)
        cross = cross.mean(0).unsqueeze(0)
        plus = plus.mean(0).unsqueeze(0)
        return cross, plus