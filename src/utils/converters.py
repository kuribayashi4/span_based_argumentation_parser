import chainer
from chainer import cuda
import numpy as np
from chainer.dataset import to_device


def convert_hybrid(batch, gpu):

    def to_device_batch(batch, gpu):
        if gpu is None:
            return batch
        elif gpu < 0:
            return [chainer.dataset.to_device(gpu, x) for x in batch]
        else:
            xp = cuda.cupy.get_array_module(*batch)
            concat = xp.concatenate(batch, axis=0)
            sections = np.cumsum([len(x) for x in batch[:-1]], dtype='i')
            concat_dev = chainer.dataset.to_device(gpu, concat)
            batch_dev = cuda.cupy.split(concat_dev, sections)
            return batch_dev

    return {'ids': [x1 for x1, t1, t2, t3 in batch],
            'ts_link': to_device(x=np.array([t1 for _, t1, t2, t3 in batch], dtype='i'), device=gpu),
            'ts_type': to_device(x=np.array([t2 for _, t1, t2, t3 in batch], dtype='i'), device=gpu),
            'ts_link_type': to_device(x=np.array([t3 for _, t1, t2, t3 in batch], dtype='i'), device=gpu)
            }
