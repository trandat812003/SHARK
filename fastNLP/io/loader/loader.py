__all__ = [
    "Loader"
]

from typing import Union, Dict

from fastNLP.io.data_bundle import DataBundle
from fastNLP.io.file_utils import _get_dataset_url, get_cache_path, cached_path
from fastNLP.io.utils import check_loader_paths
from fastNLP.core.dataset import DataSet


class Loader:
    r"""
    各种数据 **Loader** 的基类，提供了 API 的参考。
    :class:`Loader` 支持以下的三个函数

    - :meth:`download` 函数：自动将该数据集下载到缓存地址，默认缓存地址为 ``~/.fastNLP/datasets/`` 。由于版权等原因，不是所有的 ``Loader`` 都实现了该方法。
      该方法会返回下载后文件所处的缓存地址。
    - :meth:`_load` 函数：从一个数据文件中读取数据，返回一个 :class:`~fastNLP.core.DataSet` 。返回的 DataSet 的内容可以通过每个 ``Loader`` 的文档判断出。
    - :meth:`load` 函数：将文件分别读取为 :class:`~fastNLP.core.DataSet` ，然后将多个 DataSet 放入到一个 :class:`~fastNLP.io.DataBundle` 中并返回
    """
    def __init__(self):
        pass
    
    def _load(self, path: str) -> DataSet:
        r"""
        给定一个路径，返回读取的 :class:`~fastNLP.core.DataSet` 。

        :param path: 路径
        :return: :class:`~fastNLP.core.DataSet`
        """
        raise NotImplementedError
    
    def load(self, paths: Union[str, Dict[str, str]] = None) -> DataBundle:
        r"""
        从指定一个或多个路径中的文件中读取数据，返回 :class:`~fastNLP.io.DataBundle` 。

        :param paths: 支持以下的几种输入方式：

            - ``None`` -- 先查看本地是否有缓存，如果没有则自动下载并缓存。
            - 一个目录，该目录下名称包含 ``'train'`` 的被认为是训练集，包含 ``'test'`` 的被认为是测试集，包含 ``'dev'`` 的被认为是验证集 / 开发集，
              如果检测到多个文件名包含 ``'train'``、 ``'dev'``、 ``'test'`` 则会报错::

                data_bundle = xxxLoader().load('/path/to/dir')  # 返回的DataBundle中datasets根据目录下是否检测到train
                #  dev、 test 等有所变化，可以通过以下的方式取出 DataSet
                tr_data = data_bundle.get_dataset('train')
                te_data = data_bundle.get_dataset('test')  # 如果目录下有文件包含test这个字段

            - 传入一个 :class:`dict` ，比如训练集、验证集和测试集不在同一个目录下，或者名称中不包含 ``'train'``、 ``'dev'``、 ``'test'`` ::

                paths = {'train':"/path/to/tr.conll", 'dev':"/to/validate.conll", "test":"/to/te.conll"}
                data_bundle = xxxLoader().load(paths)  # 返回的DataBundle中的dataset中包含"train", "dev", "test"
                dev_data = data_bundle.get_dataset('dev')

            - 传入文件路径::

                data_bundle = xxxLoader().load("/path/to/a/train.conll") # 返回DataBundle对象, datasets中仅包含'train'
                tr_data = data_bundle.get_dataset('train')  # 取出DataSet

        :return: :class:`~fastNLP.io.DataBundle`
        """
        if paths is None:
            paths = self.download()
        paths = check_loader_paths(paths)
        datasets = {name: self._load(path) for name, path in paths.items()}
        data_bundle = DataBundle(datasets=datasets)
        return data_bundle
    
    def download(self) -> str:
        r"""
        自动下载该数据集

        :return: 下载后解压目录
        """
        raise NotImplementedError(f"{self.__class__} cannot download data automatically.")
    
    @staticmethod
    def _get_dataset_path(dataset_name):
        r"""
        传入dataset的名称，获取读取数据的目录。如果数据不存在，会尝试自动下载并缓存（如果支持的话）

        :param str dataset_name: 数据集的名称
        :return: str, 数据集的目录地址。直接到该目录下读取相应的数据即可。
        """
        
        default_cache_path = get_cache_path()
        url = _get_dataset_url(dataset_name)
        output_dir = cached_path(url_or_filename=url, cache_dir=default_cache_path, name='dataset')
        
        return output_dir

    def set_ignore_type(self, field_name, flag):
        if self.has_field(field_name):
            self.fields[field_name]['ignore_type'] = flag
        else:
            raise KeyError(f"Field {field_name} not found in DataSet.")
