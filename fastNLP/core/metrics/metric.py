__all__ = [
    'Metric',
    'MetricBase'
]

from abc import abstractmethod

from typing import Union, List
import functools
from contextlib import contextmanager
import numpy as np

from fastNLP.core.metrics.backend import Backend, AutoBackend
from fastNLP.core.metrics.element import Element
from fastNLP.envs import is_cur_env_distributed
from fastNLP.core.log import logger


import inspect
from collections import defaultdict

import numpy as np
import torch

from fastNLP.core.utils import CheckError
from fastNLP.core.utils import CheckRes
from fastNLP.core.utils import _build_args
from fastNLP.core.utils import _check_arg_dict_list
from fastNLP.core.utils import get_func_signature
from fastNLP.core.utils import seq_lens_to_masks
from fastNLP.core.vocabulary import Vocabulary


class MetricBase(object):
    """Base class for all metrics.

    ``MetricBase`` handles validity check of its input dictionaries - ``pred_dict`` and ``target_dict``.
    ``pred_dict`` is the output of ``forward()`` or prediction function of a model.
    ``target_dict`` is the ground truth from DataSet where ``is_target`` is set ``True``.
    ``MetricBase`` will do the following type checks:

        1. whether self.evaluate has varargs, which is not supported.
        2. whether params needed by self.evaluate is not included in ``pred_dict``, ``target_dict``.
        3. whether params needed by self.evaluate duplicate in ``pred_dict``, ``target_dict``.
        4. whether params in ``pred_dict``, ``target_dict`` are not used by evaluate.(Might cause warning)

    Besides, before passing params into self.evaluate, this function will filter out params from output_dict and
    target_dict which are not used in self.evaluate. (but if **kwargs presented in self.evaluate, no filtering
    will be conducted.)
    However, in some cases where type check is not necessary, ``_fast_param_map`` will be used.

    """
    def __init__(self):
        self.param_map = {}  # key is param in function, value is input param.
        self._checked = False

    def evaluate(self, *args, **kwargs):
        raise NotImplementedError

    def _init_param_map(self, key_map=None, **kwargs):
        """Check the validity of key_map and other param map. Add these into self.param_map

        :param key_map: dict
        :param kwargs:
        :return: None
        """
        value_counter = defaultdict(set)
        if key_map is not None:
            if not isinstance(key_map, dict):
                raise TypeError("key_map must be `dict`, got {}.".format(type(key_map)))
            for key, value in key_map.items():
                if value is None:
                    self.param_map[key] = key
                    continue
                if not isinstance(key, str):
                    raise TypeError(f"key in key_map must be `str`, not `{type(key)}`.")
                if not isinstance(value, str):
                    raise TypeError(f"value in key_map must be `str`, not `{type(value)}`.")
                self.param_map[key] = value
                value_counter[value].add(key)
        for key, value in kwargs.items():
            if value is None:
                self.param_map[key] = key
                continue
            if not isinstance(value, str):
                raise TypeError(f"in {key}={value}, value must be `str`, not `{type(value)}`.")
            self.param_map[key] = value
            value_counter[value].add(key)
        for value, key_set in value_counter.items():
            if len(key_set) > 1:
                raise ValueError(f"Several parameters:{key_set} are provided with one output {value}.")

        # check consistence between signature and param_map
        func_spect = inspect.getfullargspec(self.evaluate)
        func_args = [arg for arg in func_spect.args if arg != 'self']
        for func_param, input_param in self.param_map.items():
            if func_param not in func_args:
                raise NameError(
                    f"Parameter `{func_param}` is not in {get_func_signature(self.evaluate)}. Please check the "
                    f"initialization parameters, or change its signature.")

    def get_metric(self, reset=True):
        raise NotImplemented

    def _fast_param_map(self, pred_dict, target_dict):
        """Only used as inner function. When the pred_dict, target is unequivocal. Don't need users to pass key_map.
            such as pred_dict has one element, target_dict has one element

        :param pred_dict:
        :param target_dict:
        :return: dict, if dict is not {}, pass it to self.evaluate. Otherwise do mapping.
        """
        fast_param = {}
        if len(self.param_map) == 2 and len(pred_dict) == 1 and len(target_dict) == 1:
            fast_param['pred'] = list(pred_dict.values())[0]
            fast_param['target'] = list(pred_dict.values())[0]
            return fast_param
        return fast_param

    def __call__(self, pred_dict, target_dict):
        """

        This method will call self.evaluate method.
        Before calling self.evaluate, it will first check the validity of output_dict, target_dict
            (1) whether params needed by self.evaluate is not included in output_dict,target_dict.
            (2) whether params needed by self.evaluate duplicate in pred_dict, target_dict
            (3) whether params in output_dict, target_dict are not used by evaluate.(Might cause warning)
        Besides, before passing params into self.evaluate, this function will filter out params from output_dict and
            target_dict which are not used in self.evaluate. (but if **kwargs presented in self.evaluate, no filtering
            will be conducted.)
        This function also support _fast_param_map.
        :param pred_dict: usually the output of forward or prediction function
        :param target_dict: usually features set as target..
        :return:
        """
        if not callable(self.evaluate):
            raise TypeError(f"{self.__class__.__name__}.evaluate has to be callable, not {type(self.evaluate)}.")

        fast_param = self._fast_param_map(pred_dict=pred_dict, target_dict=target_dict)
        if fast_param:
            self.evaluate(**fast_param)
            return

        if not self._checked:
            # 1. check consistence between signature and param_map
            func_spect = inspect.getfullargspec(self.evaluate)
            func_args = set([arg for arg in func_spect.args if arg != 'self'])
            for func_arg, input_arg in self.param_map.items():
                if func_arg not in func_args:
                    raise NameError(f"`{func_arg}` not in {get_func_signature(self.evaluate)}.")

            # 2. only part of the param_map are passed, left are not
            for arg in func_args:
                if arg not in self.param_map:
                    self.param_map[arg] = arg  # This param does not need mapping.
            self._evaluate_args = func_args
            self._reverse_param_map = {input_arg: func_arg for func_arg, input_arg in self.param_map.items()}

        # need to wrap inputs in dict.
        mapped_pred_dict = {}
        mapped_target_dict = {}
        duplicated = []
        for input_arg in set(list(pred_dict.keys()) + list(target_dict.keys())):
            not_duplicate_flag = 0
            if input_arg in self._reverse_param_map:
                mapped_arg = self._reverse_param_map[input_arg]
                not_duplicate_flag += 1
            else:
                mapped_arg = input_arg
            if input_arg in pred_dict:
                mapped_pred_dict[mapped_arg] = pred_dict[input_arg]
                not_duplicate_flag += 1
            if input_arg in target_dict:
                mapped_target_dict[mapped_arg] = target_dict[input_arg]
                not_duplicate_flag += 1
            if not_duplicate_flag == 3:
                duplicated.append(input_arg)

        # missing
        if not self._checked:
            check_res = _check_arg_dict_list(self.evaluate, [mapped_pred_dict, mapped_target_dict])
            # only check missing.
            # replace missing.
            missing = check_res.missing
            replaced_missing = list(missing)
            for idx, func_arg in enumerate(missing):
                # Don't delete `` in this information, nor add ``
                replaced_missing[idx] = f"{self.param_map[func_arg]}" + f"(assign to `{func_arg}` " \
                                                                        f"in `{self.__class__.__name__}`)"

            check_res = CheckRes(missing=replaced_missing,
                                 unused=check_res.unused,
                                 duplicated=duplicated,
                                 required=check_res.required,
                                 all_needed=check_res.all_needed,
                                 varargs=check_res.varargs)

            if check_res.missing or check_res.duplicated:
                raise CheckError(check_res=check_res,
                                 func_signature=get_func_signature(self.evaluate))
        refined_args = _build_args(self.evaluate, **mapped_pred_dict, **mapped_target_dict)

        self.evaluate(**refined_args)
        self._checked = True

        return



class AccuracyMetric(MetricBase):
    """Accuracy Metric

    """
    def __init__(self, pred=None, target=None, seq_lens=None):
        super().__init__()

        self._init_param_map(pred=pred, target=target, seq_lens=seq_lens)

        self.total = 0
        self.acc_count = 0

    def _fast_param_map(self, pred_dict, target_dict):
        """Only used as inner function. When the pred_dict, target is unequivocal. Don't need users to pass key_map.
            such as pred_dict has one element, target_dict has one element

        :param pred_dict:
        :param target_dict:
        :return: dict, if dict is not None, pass it to self.evaluate. Otherwise do mapping.
        """
        fast_param = {}
        targets = list(target_dict.values())
        if len(targets) == 1 and isinstance(targets[0], torch.Tensor):
            if len(pred_dict) == 1:
                pred = list(pred_dict.values())[0]
                fast_param['pred'] = pred
            elif len(pred_dict) == 2:
                pred1 = list(pred_dict.values())[0]
                pred2 = list(pred_dict.values())[1]
                if not (isinstance(pred1, torch.Tensor) and isinstance(pred2, torch.Tensor)):
                    return fast_param
                if len(pred1.size()) < len(pred2.size()) and len(pred1.size()) == 1:
                    seq_lens = pred1
                    pred = pred2
                elif len(pred1.size()) > len(pred2.size()) and len(pred2.size()) == 1:
                    seq_lens = pred2
                    pred = pred1
                else:
                    return fast_param
                fast_param['pred'] = pred
                fast_param['seq_lens'] = seq_lens
            else:
                return fast_param
            fast_param['target'] = targets[0]
        # TODO need to make sure they all have same batch_size
        return fast_param

    def evaluate(self, pred, target, seq_lens=None):
        """

        :param pred: List of (torch.Tensor, or numpy.ndarray). Element's shape can be:
                torch.Size([B,]), torch.Size([B, n_classes]), torch.Size([B, max_len]), torch.Size([B, max_len, n_classes])
        :param target: List of (torch.Tensor, or numpy.ndarray). Element's can be:
                torch.Size([B,]), torch.Size([B,]), torch.Size([B, max_len]), torch.Size([B, max_len])
        :param seq_lens: List of (torch.Tensor, or numpy.ndarray). Element's can be:
                None, None, torch.Size([B], torch.Size([B]). ignored if masks are provided.

        """
        # TODO 这里报错需要更改，因为pred是啥用户并不知道。需要告知用户真实的value
        if not isinstance(pred, torch.Tensor):
            raise TypeError(f"`pred` in {get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(pred)}.")
        if not isinstance(target, torch.Tensor):
            raise TypeError(f"`target` in {get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(target)}.")

        if seq_lens is not None and not isinstance(seq_lens, torch.Tensor):
            raise TypeError(f"`seq_lens` in {get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(seq_lens)}.")

        if seq_lens is not None:
            masks = seq_lens_to_masks(seq_lens=seq_lens, float=True)
        else:
            masks = None

        if pred.size() == target.size():
            pass
        elif len(pred.size()) == len(target.size()) + 1:
            pred = pred.argmax(dim=-1)
        else:
            raise RuntimeError(f"In {get_func_signature(self.evaluate)}, when pred have "
                               f"size:{pred.size()}, target should have size: {pred.size()} or "
                               f"{pred.size()[:-1]}, got {target.size()}.")

        pred = pred.float()
        target = target.float()

        if masks is not None:
            self.acc_count += torch.sum(torch.eq(pred, target).float() * masks.float()).item()
            self.total += torch.sum(masks.float()).item()
        else:
            self.acc_count += torch.sum(torch.eq(pred, target).float()).item()
            self.total += np.prod(list(pred.size()))


    def get_metric(self, reset=True):
        """Returns computed metric.

        :param bool reset: whether to recount next time.
        :return evaluate_result: {"acc": float}
        """
        evaluate_result = {'acc': round(self.acc_count / self.total, 6)}
        if reset:
            self.acc_count = 0
            self.total = 0
        return evaluate_result


def bmes_tag_to_spans(tags, ignore_labels=None):
    """

    :param tags: List[str],
    :param ignore_labels: List[str], 在该list中的label将被忽略
    :return: List[Tuple[str, List[int, int]]]. [(label，[start, end])]
    """
    ignore_labels = set(ignore_labels) if ignore_labels else set()

    spans = []
    prev_bmes_tag = None
    for idx, tag in enumerate(tags):
        tag = tag.lower()
        bmes_tag, label = tag[:1], tag[2:]
        if bmes_tag in ('b', 's'):
            spans.append((label, [idx, idx]))
        elif bmes_tag in ('m', 'e') and prev_bmes_tag in ('b', 'm') and label==spans[-1][0]:
            spans[-1][1][1] = idx
        else:
            spans.append((label, [idx, idx]))
        prev_bmes_tag = bmes_tag
    return [(span[0], (span[1][0], span[1][1]))
                    for span in spans
                        if span[0] not in ignore_labels
            ]


def bio_tag_to_spans(tags, ignore_labels=None):
    """

    :param tags: List[str],
    :param ignore_labels: List[str], 在该list中的label将被忽略
    :return: List[Tuple[str, List[int, int]]]. [(label，[start, end])]
    """
    ignore_labels = set(ignore_labels) if ignore_labels else set()

    spans = []
    prev_bio_tag = None
    for idx, tag in enumerate(tags):
        tag = tag.lower()
        bio_tag, label = tag[:1], tag[2:]
        if bio_tag == 'b':
            spans.append((label, [idx, idx]))
        elif bio_tag == 'i' and prev_bio_tag in ('b', 'i') and label==spans[-1][0]:
            spans[-1][1][1] = idx
        elif bio_tag == 'o': # o tag does not count
            pass
        else:
            spans.append((label, [idx, idx]))
        prev_bio_tag = bio_tag
    return [(span[0], (span[1][0], span[1][1]))
                    for span in spans
                        if span[0] not in ignore_labels
            ]



class SpanFPreRecMetric(MetricBase):
    """
    在序列标注问题中，以span的方式计算F, pre, rec.
    最后得到的metric结果为
    {
        'f': xxx, # 这里使用f考虑以后可以计算f_beta值
        'pre': xxx,
        'rec':xxx
    }
    若only_gross=False, 即还会返回各个label的metric统计值
        {
        'f': xxx,
        'pre': xxx,
        'rec':xxx,
        'f-label': xxx,
        'pre-label': xxx,
        'rec-label':xxx,
        ...
    }

    """
    def __init__(self, tag_vocab, pred=None, target=None, seq_lens=None, encoding_type='bio', ignore_labels=None,
                  only_gross=True, f_type='micro', beta=1):
        """

        :param tag_vocab: Vocabulary, 标签的vocabulary。支持的标签为"B"(没有label)；或"B-xxx"(xxx为某种label，比如POS中的NN)，
            在解码时，会将相同xxx的认为是同一个label，比如['B-NN', 'E-NN']会被合并为一个'NN'.
        :param pred: str, 用该key在evaluate()时从传入dict中取出prediction数据。 为None，则使用'pred'取数据
        :param target: str, 用该key在evaluate()时从传入dict中取出target数据。 为None，则使用'target'取数据
        :param seq_lens: str, 用该key在evaluate()时从传入dict中取出sequence length数据。为None，则使用'seq_lens'取数据。
        :param encoding_type: str, 目前支持bio, bmes
        :param ignore_labels, List[str]. 这个list中的class不会被用于计算。例如在POS tagging时传入['NN']，则不会计算'NN'这
            个label
        :param only_gross, bool. 是否只计算总的f1, precision, recall的值；如果为False，不仅返回总的f1, pre, rec, 还会返回每个
            label的f1, pre, rec
        :param f_type, str. 'micro'或'macro'. 'micro':通过先计算总体的TP，FN和FP的数量，再计算f, precision, recall; 'macro':
            分布计算每个类别的f, precision, recall，然后做平均（各类别f的权重相同）
        :param beta, float. f_beta分数，f_beta = (1 + beta^2)*(pre*rec)/(beta^2*pre + rec). 常用为beta=0.5, 1, 2. 若为0.5
            则精确率的权重高于召回率；若为1，则两者平等；若为2，则召回率权重高于精确率。
        """
        encoding_type = encoding_type.lower()
        if encoding_type not in ('bio', 'bmes'):
            raise ValueError("Only support 'bio' or 'bmes' type.")
        if not isinstance(tag_vocab, Vocabulary):
            raise TypeError("tag_vocab can only be fastNLP.Vocabulary, not {}.".format(type(tag_vocab)))
        if f_type not in ('micro', 'macro'):
            raise ValueError("f_type only supports `micro` or `macro`', got {}.".format(f_type))

        self.encoding_type = encoding_type
        if self.encoding_type == 'bmes':
            self.tag_to_span_func = bmes_tag_to_spans
        elif self.encoding_type == 'bio':
            self.tag_to_span_func = bio_tag_to_spans
        self.ignore_labels = ignore_labels
        self.f_type = f_type
        self.beta = beta
        self.beta_square = self.beta**2
        self.only_gross = only_gross

        super().__init__()
        self._init_param_map(pred=pred, target=target, seq_lens=seq_lens)

        self.tag_vocab = tag_vocab

        self._true_positives = defaultdict(int)
        self._false_positives = defaultdict(int)
        self._false_negatives = defaultdict(int)

    def evaluate(self, pred, target, seq_lens):
        """
        A lot of design idea comes from allennlp's measure
        :param pred:
        :param target:
        :param seq_lens:
        :return:
        """
        if not isinstance(pred, torch.Tensor):
            raise TypeError(f"`pred` in {get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(pred)}.")
        if not isinstance(target, torch.Tensor):
            raise TypeError(f"`target` in {get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(target)}.")

        if not isinstance(seq_lens, torch.Tensor):
            raise TypeError(f"`seq_lens` in {get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(seq_lens)}.")

        if pred.size() == target.size() and len(target.size()) == 2:
            pass
        elif len(pred.size()) == len(target.size()) + 1 and len(target.size()) == 2:
            pred = pred.argmax(dim=-1)
            num_classes = pred.size(-1)
            if (target >= num_classes).any():
                raise ValueError("A gold label passed to SpanBasedF1Metric contains an "
                                 "id >= {}, the number of classes.".format(num_classes))
        else:
            raise RuntimeError(f"In {get_func_signature(self.evaluate)}, when pred have "
                               f"size:{pred.size()}, target should have size: {pred.size()} or "
                               f"{pred.size()[:-1]}, got {target.size()}.")

        batch_size = pred.size(0)
        for i in range(batch_size):
            pred_tags = pred[i, :seq_lens[i]].tolist()
            gold_tags = target[i, :seq_lens[i]].tolist()

            pred_str_tags = [self.tag_vocab.to_word(tag) for tag in pred_tags]
            gold_str_tags = [self.tag_vocab.to_word(tag) for tag in gold_tags]

            pred_spans = self.tag_to_span_func(pred_str_tags, ignore_labels=self.ignore_labels)
            gold_spans = self.tag_to_span_func(gold_str_tags, ignore_labels=self.ignore_labels)

            for span in pred_spans:
                if span in gold_spans:
                    self._true_positives[span[0]] += 1
                    gold_spans.remove(span)
                else:
                    self._false_positives[span[0]] += 1
            for span in gold_spans:
                self._false_negatives[span[0]] += 1


    def get_metric(self, reset=True):
        evaluate_result = {}
        if not self.only_gross or self.f_type=='macro':
            tags = set(self._false_negatives.keys())
            tags.update(set(self._false_positives.keys()))
            tags.update(set(self._true_positives.keys()))
            f_sum = 0
            pre_sum = 0
            rec_sum = 0
            for tag in tags:
                tp = self._true_positives[tag]
                fn = self._false_negatives[tag]
                fp = self._false_positives[tag]
                f, pre, rec = self._compute_f_pre_rec(tp, fn, fp)
                f_sum += f
                pre_sum += pre
                rec_sum + rec
                if not self.only_gross and tag!='': # tag!=''防止无tag的情况
                    f_key = 'f-{}'.format(tag)
                    pre_key = 'pre-{}'.format(tag)
                    rec_key = 'rec-{}'.format(tag)
                    evaluate_result[f_key] = f
                    evaluate_result[pre_key] = pre
                    evaluate_result[rec_key] = rec

            if self.f_type == 'macro':
                evaluate_result['f'] = f_sum/len(tags)
                evaluate_result['pre'] = pre_sum/len(tags)
                evaluate_result['rec'] = rec_sum/len(tags)

        if self.f_type == 'micro':
            f, pre, rec = self._compute_f_pre_rec(sum(self._true_positives.values()),
                                                  sum(self._false_negatives.values()),
                                                  sum(self._false_positives.values()))
            evaluate_result['f'] = f
            evaluate_result['pre'] = pre
            evaluate_result['rec'] = rec

        if reset:
            self._true_positives = defaultdict(int)
            self._false_positives = defaultdict(int)
            self._false_negatives = defaultdict(int)

        return evaluate_result

    def _compute_f_pre_rec(self, tp, fn, fp):
        """

        :param tp: int, true positive
        :param fn: int, false negative
        :param fp: int, false positive
        :return: (f, pre, rec)
        """
        pre = tp / (fp + tp + 1e-13)
        rec = tp / (fn + tp + 1e-13)
        f = (1 + self.beta_square) * pre * rec / (self.beta_square * pre + rec + 1e-13)

        return f, pre, rec


class BMESF1PreRecMetric(MetricBase):
    """
    按照BMES标注方式计算f1, precision, recall。由于可能存在非法tag，比如"BS"，所以需要用以下的表格做转换，cur_B意思是当前tag是B，
        next_B意思是后一个tag是B。则cur_B=S，即将当前被predict是B的tag标为S；next_M=B, 即将后一个被predict是M的tag标为B
        |       |  next_B |  next_M  |  next_E  |  next_S |   end   |
        |:-----:|:-------:|:--------:|:--------:|:-------:|:-------:|
        | start |   合法  | next_M=B | next_E=S |   合法  |    -    |
        | cur_B | cur_B=S |   合法   |   合法   | cur_B=S | cur_B=S |
        | cur_M | cur_M=E |   合法   |   合法   | cur_M=E | cur_M=E |
        | cur_E |   合法  | next_M=B | next_E=S |   合法  |   合法  |
        | cur_S |   合法  | next_M=B | next_E=S |   合法  |   合法  |
    举例：
        prediction为BSEMS，会被认为是SSSSS.

    本Metric不检验target的合法性，请务必保证target的合法性。
        pred的形状应该为(batch_size, max_len) 或 (batch_size, max_len, 4)。
        target形状为 (batch_size, max_len)
        seq_lens形状为 (batch_size, )

    """

    def __init__(self, b_idx=0, m_idx=1, e_idx=2, s_idx=3, pred=None, target=None, seq_lens=None):
        """
        需要申明BMES这四种tag中，各种tag对应的idx。所有不为b_idx, m_idx, e_idx, s_idx的数字都认为是s_idx。

        :param b_idx: int, Begin标签所对应的tag idx.
        :param m_idx: int, Middle标签所对应的tag idx.
        :param e_idx: int, End标签所对应的tag idx.
        :param s_idx: int, Single标签所对应的tag idx
        :param pred: str, 用该key在evaluate()时从传入dict中取出prediction数据。 为None，则使用'pred'取数据
        :param target: str, 用该key在evaluate()时从传入dict中取出target数据。 为None，则使用'target'取数据
        :param seq_lens: str, 用该key在evaluate()时从传入dict中取出seqence length数据。为None，则使用'seq_lens'取数据。
        """
        super().__init__()

        self._init_param_map(pred=pred, target=target, seq_lens=seq_lens)

        self.yt_wordnum = 0
        self.yp_wordnum = 0
        self.corr_num = 0

        self.b_idx = b_idx
        self.m_idx = m_idx
        self.e_idx = e_idx
        self.s_idx = s_idx
        # 还原init处介绍的矩阵
        self._valida_matrix = {
            -1: [(-1, -1), (1, self.b_idx), (1, self.s_idx), (-1, -1)], # magic start idx
            self.b_idx:[(0, self.s_idx), (-1, -1), (-1, -1), (0, self.s_idx), (0, self.s_idx)],
            self.m_idx:[(0, self.e_idx), (-1, -1), (-1, -1), (0, self.e_idx), (0, self.e_idx)],
            self.e_idx:[(-1, -1), (1, self.b_idx), (1, self.s_idx), (-1, -1), (-1, -1)],
            self.s_idx:[(-1, -1), (1, self.b_idx), (1, self.s_idx), (-1, -1), (-1, -1)],
        }

    def _validate_tags(self, tags):
        """
        给定一个tag的Tensor，返回合法tag

        :param tags: Tensor, shape: (seq_len, )
        :return: 返回修改为合法tag的list
        """
        assert len(tags)!=0
        assert isinstance(tags, torch.Tensor) and len(tags.size())==1
        padded_tags = [-1, *tags.tolist(), -1]
        for idx in range(len(padded_tags)-1):
            cur_tag = padded_tags[idx]
            if cur_tag not in self._valida_matrix:
                cur_tag = self.s_idx
            if padded_tags[idx+1] not in self._valida_matrix:
                padded_tags[idx+1] = self.s_idx
            next_tag = padded_tags[idx+1]
            shift_tag = self._valida_matrix[cur_tag][next_tag]
            if shift_tag[0]!=-1:
                padded_tags[idx+shift_tag[0]] = shift_tag[1]

        return padded_tags[1:-1]

    def evaluate(self, pred, target, seq_lens):
        if not isinstance(pred, torch.Tensor):
            raise TypeError(f"`pred` in {get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(pred)}.")
        if not isinstance(target, torch.Tensor):
            raise TypeError(f"`target` in {get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(target)}.")

        if not isinstance(seq_lens, torch.Tensor):
            raise TypeError(f"`seq_lens` in {get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(seq_lens)}.")

        if pred.size() == target.size() and len(target.size()) == 2:
            pass
        elif len(pred.size()) == len(target.size()) + 1 and len(target.size()) == 2:
            pred = pred.argmax(dim=-1)
        else:
            raise RuntimeError(f"In {get_func_signature(self.evaluate)}, when pred have "
                               f"size:{pred.size()}, target should have size: {pred.size()} or "
                               f"{pred.size()[:-1]}, got {target.size()}.")

        for idx in range(len(pred)):
            seq_len = seq_lens[idx]
            target_tags = target[idx][:seq_len].tolist()
            pred_tags = pred[idx][:seq_len]
            pred_tags = self._validate_tags(pred_tags)
            start_idx = 0
            for t_idx, (t_tag, p_tag) in enumerate(zip(target_tags, pred_tags)):
                if t_tag in (self.s_idx,  self.e_idx):
                    self.yt_wordnum += 1
                    corr_flag = True
                    for i in range(start_idx, t_idx+1):
                        if target_tags[i]!=pred_tags[i]:
                            corr_flag = False
                    if corr_flag:
                        self.corr_num += 1
                    start_idx = t_idx + 1
                if p_tag in (self.s_idx, self.e_idx):
                    self.yp_wordnum += 1

    def get_metric(self, reset=True):
        P = self.corr_num / (self.yp_wordnum + 1e-12)
        R = self.corr_num / (self.yt_wordnum + 1e-12)
        F = 2 * P * R / (P + R + 1e-12)
        evaluate_result = {'f': round(F, 6), 'pre':round(P, 6), 'rec': round(R, 6)}
        if reset:
            self.yp_wordnum = 0
            self.yt_wordnum = 0
            self.corr_num = 0
        return evaluate_result



def _prepare_metrics(metrics):
    """

    Prepare list of Metric based on input
    :param metrics:
    :return: List[fastNLP.MetricBase]
    """
    _metrics = []
    if metrics:
        if isinstance(metrics, list):
            for metric in metrics:
                if isinstance(metric, type):
                    metric = metric()
                if isinstance(metric, MetricBase):
                    metric_name = metric.__class__.__name__
                    if not callable(metric.evaluate):
                        raise TypeError(f"{metric_name}.evaluate must be callable, got {type(metric.evaluate)}.")
                    if not callable(metric.get_metric):
                        raise TypeError(f"{metric_name}.get_metric must be callable, got {type(metric.get_metric)}.")
                    _metrics.append(metric)
                else:
                    raise TypeError(
                        f"The type of metric in metrics must be `fastNLP.MetricBase`, not `{type(metric)}`.")
        elif isinstance(metrics, MetricBase):
            _metrics = [metrics]
        else:
            raise TypeError(f"The type of metrics should be `list[fastNLP.MetricBase]` or `fastNLP.MetricBase`, "
                            f"got {type(metrics)}.")
    return _metrics


def accuracy_topk(y_true, y_prob, k=1):
    """Compute accuracy of y_true matching top-k probable labels in y_prob.

    :param y_true: ndarray, true label, [n_samples]
    :param y_prob: ndarray, label probabilities, [n_samples, n_classes]
    :param k: int, k in top-k
    :returns acc: accuracy of top-k

    """
    y_pred_topk = np.argsort(y_prob, axis=-1)[:, -1:-k - 1:-1]
    y_true_tile = np.tile(np.expand_dims(y_true, axis=1), (1, k))
    y_match = np.any(y_pred_topk == y_true_tile, axis=-1)
    acc = np.sum(y_match) / y_match.shape[0]
    return acc



def pred_topk(y_prob, k=1):
    """Return top-k predicted labels and corresponding probabilities.

    :param y_prob: ndarray, size [n_samples, n_classes], probabilities on labels
    :param k: int, k of top-k
    :returns (y_pred_topk, y_prob_topk):
        y_pred_topk: ndarray, size [n_samples, k], predicted top-k labels
        y_prob_topk: ndarray, size [n_samples, k], probabilities for top-k labels

    """
    y_pred_topk = np.argsort(y_prob, axis=-1)[:, -1:-k - 1:-1]
    x_axis_index = np.tile(
        np.arange(len(y_prob))[:, np.newaxis],
        (1, k))
    y_prob_topk = y_prob[x_axis_index, y_pred_topk]
    return y_pred_topk, y_prob_topk


class Metric:
    """
    **fastNLP** 中 :class:`Metric` 的基类，自定义 :class:`Metric` 时，请继承该对象。使用该对象，将有助于减少在分布式状态下的 Metric 计算。

    .. note::

        在多卡情况下，所有 **fastNLP** 提供的 :class:`Metric` 默认情况下都会最终将所有设备上的评估结果集中到同一张卡上，并以此为基础输出最终的
        评测分数。如果您不需要这一功能，请将 ``aggregate_when_get_metric`` 置为 ``False`` 。

    .. note::

        如果您需要自定义自己的 :class:`Metric` ，并且有分布式训练的需求，请确保：
 
            1. 调用 :meth:`~Metric.register_element` 函数来注册需要 gather 的张量 
            2. 或在 :meth:`~Metric.get_metric` 函数中调用 :meth:`~Metric.all_gather_object` 函数来手动收集不同设备上的数据。

    :param backend: 目前支持五种类型的 backend, ``['torch', 'paddle', 'jittor', 'oneflow', 'auto']``。其中 ``'auto'`` 表示根据实际调用 :meth:`update`
        函数时传入的参数决定具体的 backend ，大部分情况下直接使用 ``'auto'`` 即可。
    :param aggregate_when_get_metric: 在计算 metric 的时候是否自动将各个进程上的相同的 element 的数字聚合后再得到 metric，
        当 backend 不支持分布式时，该参数无意义。如果为 ``None`` ，将在 :class:`~fastNLP.core.controllers.Evaluator` 中根据
        sampler 是否使用分布式进行自动设置。
    """
    def __init__(self, backend: Union[str, Backend, None] = 'auto', aggregate_when_get_metric: bool = None):
        self.backend = AutoBackend(backend)
        self._updated = False
        self.get_metric = self._sync_get_metric(self.get_metric)
        self.update = self._wrap_update(self.update)
        self.reset = self._wrap_auto_reset_elements(self.reset)
        self.aggregate_when_get_metric = aggregate_when_get_metric
        self._cannot_change_element = False
        self._call_gather_object = False # 用于检查用户是否在 get_metric 中调用了 all_gather_object
        self._elements = {}

    @property
    def elements(self) -> dict:
        return self._elements

    def register_element(self, name, value: float = 0, aggregate_method=None, backend='auto') -> Element:
        """
        注册一个 element 对象，注册之后便可以通过在 Metric 中直接通过 ``self.{name}`` 进行调用，可以认为该对象即为对应 backend 的
        tensor 直接进行加减乘除计算即可。

        .. warning::

            如果想使得该 metric 可自动扩展到多卡的情况，请一定申明 ``aggregate_method`` 。

        :param name: 当前 element 的名字，注册后，在 Metric 中可以通过 ``self.{name}`` 访问该变量。
        :param value: 初始化的值。在调用 :meth:`Metric.reset` 方法时也将自动设置为该值
        :param aggregate_method: 如何聚合多卡上的结果，如果为单卡执行，该值无意义。如果设置为 None 则表示该 element 不进行聚合。
        :param backend: 使用的 backend 。Element 的类型会根据 ``backend`` 进行实际的初始化。例如 ``backend`` 为 ``'torch'`` 则该对象为
            :class:`torch.Tensor` ； 如果 ``'backend'`` 为 ``'paddle'`` 则该对象为 :class:`paddle.Tensor` ；如果 ``backend`` 为
            ``'jittor'`` , 则该对象为 :class:`jittor.Var` 。一般情况下直接默认为 ``'auto'`` 就行了， **fastNLP** 会根据实际调用 :meth`Metric.update`
            函数时传入的参数进行合理的初始化，例如当传入的参数中只包含 :class:`torch.Tensor` 这一种 tensor 时（可以有其它非 tensor 类型的输入）
            则认为 ``backend`` 为 ``'torch'`` ；只包含 :class:`jittor.Var` 这一种 tensor 时（可以有其它非 tensor 类型的输入）则认为 ``backend``
            为 ``'jittor'`` 。如果没有检测到任何一种 tensor ，就默认使用 :class:`float` 类型作为 element 。
        :return: 注册的 Element 对象
        """
        if backend == 'auto':
            backend = self.backend
        else:
            backend = AutoBackend(backend)

        assert name is not None and name not in self.elements

        element = Element(name=name, value=value, aggregate_method=aggregate_method, backend=backend)
        self.elements[name] = element
        setattr(self, name, element)
        return element

    def reset(self):
        """
        在对每个 ``evaluate_dataloaders`` 遍历进行验证之前，:meth:`reset` 函数会被调用来重置每个非 element 对象；
        如果有非 element 的对象需要重置的时候，在本方法中写下非 element 的重置方式。注册的 element 对象则会自动 reset 为初始值。
        """
        pass

    def _wrap_auto_reset_elements(self, reset):
        @functools.wraps(reset)
        def _wrap_reset(*args, **kwargs):
            self._updated = False
            for ele in self.elements.values():
                ele.reset()
            reset(*args, **kwargs)

        return _wrap_reset

    def _sync_get_metric(self, get_metric):
        @functools.wraps(get_metric)
        def _wrap_get_metric(*args, **kwargs):
            assert self._updated, f"You have to call `{self.__class__.__name__}'s update() function before calling " \
                                  f"get_metric()."
            with self.sync(recover=True, aggregate=self.aggregate_when_get_metric):
                self._call_gather_object = False
                results = get_metric(*args, **kwargs)
                
                # elements 为空、没有 call 则准备报错
                if len(self._elements) == 0 and not self._call_gather_object:
                    # 需要 aggregate 并且在多卡环境下
                    if self.aggregate_when_get_metric and is_cur_env_distributed():
                        logger.rank_zero_warning("There is no `<class 'Element'>` registered in metric `{}` and you didn't call "
                                                "`Metric.all_gather_object()` in method `get_metric()` either. Therefore your "
                                                "results may not be aggregated in distributed training."
                                                .format(self.__class__), once=True)

            return results

        return _wrap_get_metric

    def __setattr__(self, key, value):
        if getattr(self, '_cannot_change_element', False):
            if key in self.elements and isinstance(value, (float, int, bool)):
                self.elements[key].fill_value(value)
                return
            elif key in self.elements:
                raise TypeError(f"self.{key} is an Element, only float/int/bool type value can be assigned to it, "
                                f"instead of {type(value)}.")
        if isinstance(value, Element) and key not in self.elements:
            raise RuntimeError("Please use register_element() function to add Element.")
        attrs = self.__dict__
        if key in attrs and isinstance(value, Element):
            raise RuntimeError(f'`{key}` has been registered as an attribute, cannot be registered as an Element!')
        object.__setattr__(self, key, value)

    # 当调用 __getattribute__ 没有找到时才会触发这个, 保留这个的目的只是为了防止 ide 的 warning
    def __getattr__(self, name: str) -> Element:
        if 'elements' in self.__dict__:
            elements = self.__dict__['elements']
            if name in elements:
                return elements[name]
        raise AttributeError("`{}` object has no attribute `{}`.".format(type(self).__name__, name))

    def _wrap_update(self, update):
        @functools.wraps(update)
        def _wrap_update(*args, **kwargs):
            self.check_backend(*args, **kwargs)
            self._cannot_change_element = True
            self._updated = True
            return update(*args, **kwargs)

        return _wrap_update

    def _wrap_check_get_metric(self, get_metric):
        """
        统计 get_metric 函数中是否调用了 self.all_gather_object() 函数
        """
        @functools.wraps(get_metric)
        def _wrapper(*args, **kwargs):
            if self._check_get_metric or len(self._elements) != 0:
                # 已经检查过，或有 Element 成员，不进行处理
                return get_metric(*args, **kwargs)
            # 否则包裹 self.all_gather_object，统计是否进行了调用
            self._check_get_metric = True
            self._call_gather_object = False
            res = get_metric(*args, **kwargs)

            if self.aggregate_when_get_metric and not self._call_gather_object:
                # warning
                logger.warning("There is no `<class 'Element'>` registered in metric `{}` and you didn't call "
                                "`Metric.all_gather_object()` in method `get_metric()` either. This may cause "
                                "some problems in distributed training since the results are not aggregated."
                                .format(self.__class__))

            return res

        return _wrapper

    def check_backend(self, *args, **kwargs):
        """
        根据传入的参数的类型选择当前需要的 backend
        """
        if not self.backend.is_specified():
            _args = []
            for arg in args:
                _args.append(arg)
            for arg in kwargs.values():
                _args.append(arg)
            self.backend.choose_real_backend(_args)

    @contextmanager
    def sync(self, recover=True, aggregate=False):
        """
        在这个上下文下， :meth:`Metric` 会自动先同步需要同步操作的 element 。当 ``recover`` 为 ``True`` 时，在退出环境的时候，会重新将 element 的
        值恢复到计算前的值。
        """
        keep_value = {}
        if aggregate:
            for name, element in self.elements.items():
                # 保存过去的值
                keep_value[name] = element.get_scalar()
                # 聚合结果
                element.aggregate()

        yield

        if recover and aggregate:
            for name, element in self.elements.items():
                # 恢复结果
                if name in keep_value:
                    element.fill_value(value=keep_value.get(name))

    @abstractmethod
    def update(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def get_metric(self) -> dict:
        raise NotImplementedError()

    def set_auto_aggregate_when_get_metric(self, flag: bool):
        """
        设置是否在 :meth:`get_metric` 的时候自动 aggregate

        """
        self.aggregate_when_get_metric = flag

    def tensor2numpy(self, tensor) -> np.array:
        """
        将 ``tensor`` 向量转为 :class:`numpy.array` 类型变量。

        :param tensor:
        :return:
        """
        return self.backend.tensor2numpy(tensor)

    def to(self, device):
        """
        将所有的 element 变量移动到 ``device`` 设备上

        :param device:
        :return:
        """
        for element in self.elements.values():
            element.to(device)

    def all_gather_object(self, obj, group=None)->List:
        """
        给定 ``obj`` 将各个 rank 上的 ``obj`` 汇总到每个 ``obj`` 上。返回一个 list 对象，里面依次为各个 rank 对应的 ``obj`` 。

        :param obj: 需要汇总的对象，必须是个 pickable 的对象。
        :param group:
        :return: -> List[obj0, obj1, ...] 其中 obj0 是rank 0 上的 obj；obj1 是 rank 1 上的 obj...
        """
        self._call_gather_object = True
        if self.aggregate_when_get_metric:
            return self.backend.all_gather_object(obj, group=group)
        return [obj]


def _prepare_metrics(metrics):
    """

    Prepare list of Metric based on input
    :param metrics:
    :return: List[fastNLP.MetricBase]
    """
    _metrics = []
    if metrics:
        if isinstance(metrics, list):
            for metric in metrics:
                if isinstance(metric, type):
                    metric = metric()
                if isinstance(metric, MetricBase):
                    metric_name = metric.__class__.__name__
                    if not callable(metric.evaluate):
                        raise TypeError(f"{metric_name}.evaluate must be callable, got {type(metric.evaluate)}.")
                    if not callable(metric.get_metric):
                        raise TypeError(f"{metric_name}.get_metric must be callable, got {type(metric.get_metric)}.")
                    _metrics.append(metric)
                else:
                    raise TypeError(
                        f"The type of metric in metrics must be `fastNLP.MetricBase`, not `{type(metric)}`.")
        elif isinstance(metrics, MetricBase):
            _metrics = [metrics]
        else:
            raise TypeError(f"The type of metrics should be `list[fastNLP.MetricBase]` or `fastNLP.MetricBase`, "
                            f"got {type(metrics)}.")
    return _metrics
