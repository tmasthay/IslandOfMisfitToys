import torch
from deepwave import scalar
from mh.core import DotDict, DotDictImmutable
from omegaconf import OmegaConf, DictConfig
from os.path import join as pj, abspath as absp


# Generate a velocity model constrained to be within a desired range
class Model(torch.nn.Module):
    def __init__(self, initial, min_vel, max_vel):
        super().__init__()
        self.min_vel = min_vel
        self.max_vel = max_vel
        self.model = torch.nn.Parameter(
            torch.logit((initial - min_vel) / (max_vel - min_vel))
        )

    def forward(self):
        return (
            torch.sigmoid(self.model) * (self.max_vel - self.min_vel)
            + self.min_vel
        )


class Prop(torch.nn.Module):
    def __init__(
        self,
        *,
        model,
        dx,
        dt,
        freq,
        source_amplitudes,
        source_locations,
        receiver_locations
    ):
        super().__init__()
        self.model = model
        self.dx = dx
        self.dt = dt
        self.freq = freq
        self.source_amplitudes = source_amplitudes
        self.source_locations = source_locations
        self.receiver_locations = receiver_locations

    def forward(self, slicer):
        v = self.model()
        if slicer is None:
            slicer = slice(None)
        return scalar(
            v,
            self.dx,
            self.dt,
            source_amplitudes=self.source_amplitudes[slicer],
            source_locations=self.source_locations[slicer],
            receiver_locations=self.receiver_locations[slicer],
            max_vel=2500,
            pml_freq=self.freq,
            time_pad_frac=0.2,
        )


def cl_assert(value, *msg, sep='\n    '):
    full_msg = sep + sep.join(msg)
    assert value, full_msg

def assert_key(key, d):
    cl_assert(
        key not in d,
        f'"{key}" is a pre-existing key in config',
        'This is incompatible with the load_data function',
        f'Your dictionary d =\n{d}'
    )


def plain_init(c: DictConfig, *, delete_keys=None):
    delete_keys = delete_keys or []
    if not isinstance(c, DotDict):
        c = DotDict(OmegaConf.to_container(c, resolve=True))
    for key in delete_keys:
        del c[key]
    return c


def load_data(c: DictConfig, *, fields: dict, delete_keys=None, path=None):
    d = plain_init(c, delete_keys=delete_keys)

    assert_key('data', d)
    assert_key('meta', d)

    def getp(x, ext='.pt'):
        x = x.replace(ext, '') + ext
        if path is None:
            return absp(x)
        else:
            return pj(path, x)
        
    d.data = DotDict()
    for field, lcl_path in fields.items():
        if field == 'meta':
            d.meta = DotDict(eval(open(getp(lcl_path, '.pydict'), 'r').read()))
            if 'derived' in d.meta:
                del d.meta.derived
        else:
            d.data[field] = torch.load(getp(lcl_path))
    return d

def preprocess_data(c, *, callbacks, delete_keys=None):
    delete_keys = delete_keys or []
    callbacks = callbacks or {}
    data = c.data
    cl_assert(set(callbacks.keys()).issubset(set(data.keys())),
              f'callbacks keys must be subset of data, {callbacks.keys()=}, {data.keys()=}')
    for field, callback in callbacks.items():
        if callback:
            data[field] = callback(field)
    for key in delete_keys:
        del c[key]
    return c  

        
    
