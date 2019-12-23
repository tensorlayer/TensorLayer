class Dataset(object):

    def __getitem__(self, index):
        raise NotImplementedError("A Dataset must implement __getitem__(index) method.")

    def __len__(self):
        raise NotImplementedError("A Dataset must implement __len__() method.")

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def __call__(self, *args, **kwargs):
        return self.__iter__()


class DatasetWrapper(object):
    def __init__(self, ds):
        self.ds = ds
        self.ds_len = len(ds)

    def __len__(self):
        return len(self.ds)

    def __iter__(self):
        for dp in self.ds:
            yield dp

    def __call__(self, *args, **kwargs):
        return self.__iter__()


class IndexableDatasetWrapper(object):
    def __init__(self, ds):
        self.ds = ds
        self.ds_len = len(ds)

    def __getitem__(self, index):
        return self.ds.__getitem__(index)

    def __len__(self):
        return len(self.ds)

    def __call__(self, *args, **kwargs):
        return self


class Transform(object):
    def __call__(self, *args, **kwargs):
        raise NotImplementedError("Transform must implement __call__() method.")


class _Transforms_for_tf_dataset(object):
    """
    This class aggregate Transforms into one object in order to use tf.data.Dataset.map API
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args):
        data_list = list(args)
        for transform in self.transforms:
            data_list = transform(*data_list)
        return data_list
