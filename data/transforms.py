'Custom transforms extending MONAI transforms'
import cv2
import monai
import numpy as np



class LoadPngImage(monai.transforms.Transform):
    'Output selected slice from input volume.'
    def __call__(self, data):
        img = cv2.imread(data, cv2.IMREAD_GRAYSCALE)
        return img


class LoadPngImaged(monai.transforms.Transform):
    'Output selected slice from input volume.'
    def __init__(self, keys):
        self.backend = LoadPngImage()
        self.keys = keys

    def __call__(self, data):
        d = dict(data)
        for k in d.keys():
            if k not in self.keys:
                continue
            d[k] = self.backend.__call__(d[k])

        return d


class LoadNumpyImage(monai.transforms.Transform):
    'Output selected slice from input volume.'
    def __call__(self, data):
        img = np.load(data)
        return img


class LoadNumpyImaged(monai.transforms.Transform):
    'Output selected slice from input volume.'
    def __init__(self, keys):
        self.backend = LoadNumpyImage()
        self.keys = keys

    def __call__(self, data):
        d = dict(data)
        for k in d.keys():
            if k not in self.keys:
                continue
            d[k] = self.backend.__call__(d[k])

        return d


class SliceImage(monai.transforms.Transform):
    'Output selected slice from input volume.'
    def __init__(self, slice_pos, time_frame):
        self.slice_pos = slice_pos
        self.time_frame = time_frame

    def __call__(self, data):
        # return data[..., self.slice_pos, self.time_frame]
        return data[45:-45, :, self.slice_pos, self.time_frame]


class Slice2DImage(monai.transforms.Transform):
    'Output selected slice from input volume.'
    def __call__(self, data):
        return data[45:-45, :].squeeze()


class SliceImageMid(monai.transforms.Transform):
    'Output selected slice from input volume.'
    def __init__(self, time_frame):
        self.time_frame = time_frame

    def __call__(self, data):
        mid = data.shape[2]
        # return data[..., mid//2, self.time_frame]
        return data[45:-45, :, mid//2, self.time_frame]
        # return data[40:-40, :-30, mid//2, self.time_frame]


class SliceImageZ(monai.transforms.Transform):
    'Output selected slice from input volume.'
    def __init__(self, slice_pos):
        self.slice_pos = slice_pos

    def __call__(self, data):
        return data[45:-45, :, self.slice_pos, :]
        # return data[..., self.slice_pos, :]


class SliceImaged(monai.transforms.Transform):
    'Output selected slice from input volume.'
    def __init__(self, keys, slice_pos, time_frame):
        self.backend = SliceImage(slice_pos, time_frame)
        self.keys = keys
        self.slice_pos = slice_pos
        self.time_frame = time_frame

    def __call__(self, data):
        d = dict(data)
        for k in d.keys():
            if k not in self.keys:
                continue
            d[k] = self.backend.__call__(d[k])

        return d


class Slice2DImaged(monai.transforms.Transform):
    'Output selected slice from input volume.'
    def __init__(self, keys):
        self.backend = Slice2DImage()
        self.keys = keys

    def __call__(self, data):
        d = dict(data)
        for k in d.keys():
            if k not in self.keys:
                continue
            d[k] = self.backend.__call__(d[k])

        return d
