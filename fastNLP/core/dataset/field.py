r"""
.. todo::
    doc
"""
__all__ = [
    'FieldArray'
]

from collections import Counter
from typing import Any, Union, List, Callable
from ..log import logger

import numpy as np


class PadderBase:
    """
        所有padder都需要继承这个类，并覆盖__call__()方法。
        用于对batch进行padding操作。传入的element是inplace的，即直接修改element可能导致数据变化，建议inplace修改之前deepcopy一份。
    """
    def __init__(self, pad_val=0, **kwargs):
        self.pad_val = pad_val

    def set_pad_val(self, pad_val):
        self.pad_val = pad_val

    def __call__(self, contents, field_name, field_ele_dtype):
        """
        传入的是List内容。假设有以下的DataSet。
        from fastNLP import DataSet
        from fastNLP import Instance
        dataset = DataSet()
        dataset.append(Instance(word='this is a demo', length=4,
                                    chars=[['t', 'h', 'i', 's'], ['i', 's'], ['a'], ['d', 'e', 'm', 'o']]))
        dataset.append(Instance(word='another one', length=2,
                                    chars=[['a', 'n', 'o', 't', 'h', 'e', 'r'], ['o', 'n', 'e']]))
        # 如果batch_size=2, 下面只是用str的方式看起来更直观一点，但实际上可能word和chars在pad时都已经为index了。
        word这个field的pad_func会接收到的内容会是
            [
                'this is a demo',
                'another one'
            ]
        length这个field的pad_func会接收到的内容会是
            [4, 2]
        chars这个field的pad_func会接收到的内容会是
            [
                [['t', 'h', 'i', 's'], ['i', 's'], ['a'], ['d', 'e', 'm', 'o']],
                [['a', 'n', 'o', 't', 'h', 'e', 'r'], ['o', 'n', 'e']]
            ]
        即把每个instance中某个field的内容合成一个List传入
        :param contents: List[element]。传入的element是inplace的，即直接修改element可能导致数据变化，建议inplace修改之前
            deepcopy一份。
        :param field_name: str, field的名称，帮助定位错误
        :param field_ele_dtype: np.int64, np.float64, np.str. 该field的内层list元素的类型。辅助判断是否pad，大多数情况用不上
        :return: List[padded_element]或np.array([padded_element])
        """
        raise NotImplementedError



class AutoPadder(PadderBase):
    """
    根据contents的数据自动判定是否需要做padding。
    (1) 如果元素类型(元素类型是指field中最里层List的元素的数据类型, 可以通过FieldArray.dtype查看，比如['This', 'is', ...]的元素类
        型为np.str, [[1,2], ...]的元素类型为np.int64)的数据不为(np.int64, np.float64)则不会进行padding
    (2) 如果元素类型为(np.int64, np.float64),
        (2.1) 如果该field的内容只有一个，比如为sequence_length, 则不进行padding
        (2.2) 如果该field的内容为List, 那么会将Batch中的List pad为一样长。若该List下还有里层的List需要padding，请使用其它padder。
            如果某个instance中field为[1, 2, 3]，则可以pad； 若为[[1,2], [3,4, ...]]则不能进行pad
    """
    def __init__(self, pad_val=0):
        """
        :param pad_val: int, padding的位置使用该index
        """
        super().__init__(pad_val=pad_val)

    def _is_two_dimension(self, contents):
        """
        判断contents是不是只有两个维度。[[1,2], [3]]是两个维度. [[[1,2], [3, 4, 5]], [[4,5]]]有三个维度
        :param contents:
        :return:
        """
        value = contents[0]
        if isinstance(value , (np.ndarray, list)):
            value = value[0]
            if isinstance(value, (np.ndarray, list)):
                return False
            return True
        return False

    def __call__(self, contents, field_name, field_ele_dtype):
        if not is_iterable(contents[0]):
            array = np.array([content for content in contents], dtype=field_ele_dtype)
        elif field_ele_dtype in (np.int64, np.float64) and self._is_two_dimension(contents):
            max_len = max([len(content) for content in contents])
            array = np.full((len(contents), max_len), self.pad_val, dtype=field_ele_dtype)
            for i, content in enumerate(contents):
                array[i][:len(content)] = content
        else:  # should only be str
            array = np.array([content for content in contents])
        return array


class FieldArray:
    """
    :class:`~fastNLP.core.dataset.DatSet` 中用于表示列的数据类型。

    :param name: 字符串的名称
    :param content: 任意类型的数据
    """

    def __init__(self, name: str, content, padder=AutoPadder(pad_val=0), is_input=None, is_target=None):
        if len(content) == 0:
            raise RuntimeError("Empty fieldarray is not allowed.")
        _content = content
        
        try:
            _content = list(_content)
        except BaseException as e:
            logger.error(f"Cannot convert content(of type:{type(content)}) into list.")
            raise e
        self.name = name
        self.content = _content
        self.set_padder(padder)
        self._is_target = None
        self._is_input = None

        self.BASIC_TYPES = (int, float, str, np.ndarray)
        self.is_2d_list = False
        self.pytype = None  # int, float, str, or np.ndarray
        self.dtype = None  # np.int64, np.float64, np.str

        if is_input is not None:
            self.is_input = is_input
        if is_target is not None:
            self.is_target = is_target

    @property
    def is_target(self):
        return self._is_target

    @is_target.setter
    def is_target(self, value):
        if value is True:
            self.pytype = self._type_detection(self.content)
            self.dtype = self._map_to_np_type(self.pytype)
        self._is_target = value

    @property
    def is_input(self):
        return self._is_input

    @is_input.setter
    def is_input(self, value):
        if value is True:
            self.pytype = self._type_detection(self.content)
            self.dtype = self._map_to_np_type(self.pytype)
        self._is_input = value

    @staticmethod
    def _map_to_np_type(basic_type):
        type_mapping = {int: np.int64, float: np.float64, str: np.str_, np.ndarray: np.ndarray}
        return type_mapping[basic_type]

    def _type_detection(self, content):
        """

        :param content: a list of int, float, str or np.ndarray, or a list of list of one.
        :return type: one of int, float, str, np.ndarray

        """
        if isinstance(content, list) and len(content) > 0 and isinstance(content[0], list):
            # content is a 2-D list
            if not all(isinstance(_, list) for _ in content):  # strict check 2-D list
                raise TypeError("Please provide 2-D list.")
            type_set = set([self._type_detection(x) for x in content])
            if len(type_set) == 2 and int in type_set and float in type_set:
                type_set = {float}
            elif len(type_set) > 1:
                raise TypeError("Cannot create FieldArray with more than one type. Provided {}".format(type_set))
            self.is_2d_list = True
            return type_set.pop()

        elif isinstance(content, list):
            # content is a 1-D list
            if len(content) == 0:
                # the old error is not informative enough.
                raise RuntimeError("Cannot create FieldArray with an empty list. Or one element in the list is empty.")
            type_set = set([type(item) for item in content])

            # breakpoint()

            if len(type_set) == 1 and tuple(type_set)[0] in self.BASIC_TYPES:
                return type_set.pop()
            elif len(type_set) == 2 and float in type_set and int in type_set:
                # up-cast int to float
                return float
            elif len(type_set) == 2 and tuple in type_set:
                return np.ndarray
            else:
                print(self.name)
                raise TypeError("Cannot create FieldArray with type {}".format(*type_set))
        else:
            raise TypeError("Cannot create FieldArray with type {}".format(type(content)))

    def set_pad_val(self, pad_val):
        """
        修改padder的pad_val.
        :param pad_val: int。
        :return:
        """
        if self.padder is not None:
            self.padder.set_pad_val(pad_val)

    def set_padder(self, padder):
        """
        设置padding方式

        :param padder: PadderBase类型或None. 设置为None即删除padder.
        :return:
        """
        if padder is not None:
            assert isinstance(padder, PadderBase), "padder must be of type PadderBase."
        self.padder = padder

    def append(self, val: Any) -> None:
        r"""
        :param val: 把该 ``val`` 添加到 fieldarray 中。
        """
        self.content.append(val)

    def pop(self, index: int) -> None:
        r"""
        删除该 field 中 ``index`` 处的元素

        :param index: 从 ``0`` 开始的数据下标。
        """
        self.content.pop(index)

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]

    def __getitem__(self, indices: Union[int, List[int]]):
        return self.get(indices)

    def __setitem__(self, idx: int, val: Any):
        assert isinstance(idx, int)
        self.content[idx] = val

    def get(self, indices: Union[int, List[int]]):
        r"""
        根据给定的 ``indices`` 返回内容。

        :param indices: 获取 ``indices`` 对应的内容。
        :return: 根据给定的 ``indices`` 返回的内容，可能是单个值 或 :class:`numpy.ndarray`
        """
        if isinstance(indices, int):
            if indices == -1:
                indices = len(self) - 1
            assert 0 <= indices < len(self)
            return self.content[indices]

        try:
            contents = [self.content[i] for i in indices]
        except BaseException as e:
            raise e

        # Kiểm tra và xử lý các phần tử không đồng nhất
        if any(isinstance(i, (list, np.ndarray)) for i in contents):
            max_len = max(len(item) if isinstance(item, (list, np.ndarray)) else 1 for item in contents)
            uniform_contents = []
            for item in contents:
                if isinstance(item, (list, np.ndarray)):
                    if len(item) < max_len:
                        # Pad the item with zeros to match the max length
                        padded_item = list(item) + [0] * (max_len - len(item))
                        uniform_contents.append(padded_item)
                    else:
                        uniform_contents.append(item)
                else:
                    # Pad single values to match the max length
                    uniform_contents.append([item] + [0] * (max_len - 1))
            return np.array(uniform_contents)
        else:
            return np.array(contents)

    def __len__(self):
        r"""
        返回长度

        :return:
        """
        return len(self.content)

    def split(self, sep: str = None, inplace: bool = True):
        r"""
        依次对自身的元素使用 ``.split()`` 方法，应该只有当本 field 的元素为 :class:`str` 时，该方法才有用。

        :param sep: 分割符，如果为 ``None`` 则直接调用 ``str.split()``。
        :param inplace: 如果为 ``True``，则将新生成值替换本 field。否则返回 :class:`list`。
        :return: List[List[str]] or self
        """
        new_contents = []
        for index, cell in enumerate(self.content):
            try:
                new_contents.append(cell.split(sep))
            except Exception as e:
                logger.error(f"Exception happens when process value in index {index}.")
                raise e
        return self._after_process(new_contents, inplace=inplace)

    def int(self, inplace: bool = True):
        r"""
        将本 field 中的值调用 ``int(cell)``. 支持 field 中内容为以下两种情况:

            * ['1', '2', ...](即 field 中每个值为 :class:`str` 的)，
            * [['1', '2', ..], ['3', ..], ...](即 field 中每个值为一个 :class:`list` ，:class:`list` 中的值会被依次转换。)

        :param inplace: 如果为 ``True``，则将新生成值替换本 field，并返回当前 field 。否则返回 :class:`list`。
        :return: List[int], List[List[int]], self
        """
        new_contents = []
        for index, cell in enumerate(self.content):
            try:
                if isinstance(cell, list):
                    new_contents.append([int(value) for value in cell])
                else:
                    new_contents.append(int(cell))
            except Exception as e:
                print(f"Exception happens when process value in index {index}.")
                raise e
        return self._after_process(new_contents, inplace=inplace)

    def float(self, inplace=True):
        r"""
        将本 field 中的值调用 ``float(cell)``. 支持 field 中内容为以下两种情况:

            * ['1', '2', ...](即 field 中每个值为 :class:`str` 的)，
            * [['1', '2', ..], ['3', ..], ...](即 field 中每个值为一个 :class:`list` ，:class:`list` 中的值会被依次转换。)

        :param inplace: 如果为 ``True``，则将新生成值替换本 field，并返回当前 field 。否则返回 :class:`list`。
        :return:
        """
        new_contents = []
        for index, cell in enumerate(self.content):
            try:
                if isinstance(cell, list):
                    new_contents.append([float(value) for value in cell])
                else:
                    new_contents.append(float(cell))
            except Exception as e:
                print(f"Exception happens when process value in index {index}.")
                raise e
        return self._after_process(new_contents, inplace=inplace)

    def bool(self, inplace=True):
        r"""
        将本field中的值调用 ``bool(cell)``. 支持 field 中内容为以下两种情况

            * ['1', '2', ...](即 field 中每个值为 :class:`str` 的)，
            * [['1', '2', ..], ['3', ..], ...](即 field 中每个值为一个 :class:`list` ，:class:`list` 中的值会被依次转换。)

        :param inplace: 如果为 ``True``，则将新生成值替换本 field，并返回当前 field 。否则返回 :class:`list`。
        :return:
        """
        new_contents = []
        for index, cell in enumerate(self.content):
            try:
                if isinstance(cell, list):
                    new_contents.append([bool(value) for value in cell])
                else:
                    new_contents.append(bool(cell))
            except Exception as e:
                print(f"Exception happens when process value in index {index}.")
                raise e

        return self._after_process(new_contents, inplace=inplace)

    def lower(self, inplace=True):
        r"""
        将本 field 中的值调用 ``cell.lower()``， 支持 field 中内容为以下两种情况

            * ['1', '2', ...](即 field 中每个值为 :class:`str` 的)，
            * [['1', '2', ..], ['3', ..], ...](即 field 中每个值为一个 :class:`list` ，:class:`list` 中的值会被依次转换。)

        :param inplace: 如果为 ``True``，则将新生成值替换本 field，并返回当前 field 。否则返回 :class:`list`。
        :return: List[int], List[List[int]], self
        """
        new_contents = []
        for index, cell in enumerate(self.content):
            try:
                if isinstance(cell, list):
                    new_contents.append([value.lower() for value in cell])
                else:
                    new_contents.append(cell.lower())
            except Exception as e:
                print(f"Exception happens when process value in index {index}.")
                raise e
        return self._after_process(new_contents, inplace=inplace)

    def upper(self, inplace=True):
        r"""
        将本 field 中的值调用 ``cell.upper()``， 支持 field 中内容为以下两种情况

            * ['1', '2', ...](即 field 中每个值为 :class:`str` 的)，
            * [['1', '2', ..], ['3', ..], ...](即 field 中每个值为一个 :class:`list` ，:class:`list` 中的值会被依次转换。)

        :param inplace: 如果为 ``True``，则将新生成值替换本 field，并返回当前 field 。否则返回 :class:`list`。
        :return: List[int], List[List[int]], self
        """
        new_contents = []
        for index, cell in enumerate(self.content):
            try:
                if isinstance(cell, list):
                    new_contents.append([value.upper() for value in cell])
                else:
                    new_contents.append(cell.upper())
            except Exception as e:
                print(f"Exception happens when process value in index {index}.")
                raise e
        return self._after_process(new_contents, inplace=inplace)

    def value_count(self) -> Counter:
        r"""
        返回该 field 下不同 value 的数量。多用于统计 label 数量

        :return: 计数结果，key 是 label，value 是出现次数
        """
        count = Counter()

        def cum(cells):
            if isinstance(cells, Callable) and not isinstance(cells, str):
                for cell_ in cells:
                    cum(cell_)
            else:
                count[cells] += 1

        for cell in self.content:
            cum(cell)
        return count

    def _after_process(self, new_contents: list, inplace: bool):
        r"""
        当调用处理函数之后，决定是否要替换 field。

        :param new_contents:
        :param inplace:
        :return: self或者生成的content
        """
        if inplace:
            self.content = new_contents
            return self
        else:
            return new_contents
