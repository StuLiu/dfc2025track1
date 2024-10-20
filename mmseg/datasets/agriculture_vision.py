import logging
import os.path
import os.path as osp
import torch
import random
import pickle
import copy
import numpy as np
import tifffile

from tqdm import tqdm
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset
from torchvision.io import read_image


@DATASETS.register_module()
class AgricultureVisionDataset(BaseSegDataset):
    METAINFO = dict(
        classes=(
            "background",
            "double_plant",
            "drydown",
            "endrow",
            "nutrient_deficiency",
            "planter_skip",
            "water",
            "waterway",
            "weed_cluster",
        ),
        palette=[
            [0, 0, 0],
            [0, 0, 63],
            [0, 63, 63],
            [0, 63, 0],
            [0, 63, 127],
            [0, 63, 191],
            [0, 63, 255],
            [0, 127, 63],
            [0, 127, 127],
        ],
    )

    def __init__(
            self,
            img_suffix=".tif",
            seg_map_suffix=".png",
            ignore_index=255,
            rare_class_sampling=False,
            temperature=0.1,
            **kwargs
    ) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            ignore_index=ignore_index,
            **kwargs
        )
        self.class_num = len(self.METAINFO['classes'])
        self.data_list = self.load_data_list()
        self._fully_initialized = True

        self.rcs = rare_class_sampling
        self.temperature = temperature
        self.class_weight, self.cid2idxes = None, {}
        if self.rcs:
            self.class_weight, self.cid2idxes = self._parse_data(self.data_list)
            self.weight_sum = self.class_weight.sum()
            print(self.class_weight)

    def _parse_data(self, data_list, save_dir=None):
        """
        parse datalist and get class weights and cid2idxes.
        Args:
            data_list: pathes for images and segmaps, List[dict{}, ...]

        Returns:
            class_weight: Tensor, List[float, ...]
            cid2idxes: dict, dict{int: [dict(), ...], ...}
        """
        # loading exist class_weight and cid2idxes
        path_class_weight = osp.join(self.data_root, 'class_weight.npy')
        path_cid2idxes = osp.join(self.data_root, 'cid2idxes.pkl')
        logging.log(level=0, msg=f'loading from {path_class_weight}, {path_cid2idxes}.')
        if os.path.exists(path_class_weight) and os.path.exists(path_cid2idxes):
            class_weight = np.load(path_class_weight)
            with open(path_cid2idxes, 'rb') as f:
                cid2idxes = pickle.load(f)
            return class_weight, cid2idxes

        # parsing
        logging.log(level=0, msg=f'parsing train set, getting class weight and cid2idxes.')
        freq_list = np.zeros([self.class_num, ]).astype(np.float32)
        cid2idxes = {cid: [] for cid in range(self.class_num)}
        for idx_, data_ in tqdm(enumerate(data_list)):
            seg_map_path = data_['seg_map_path']
            cid_list = torch.unique(read_image(seg_map_path)).numpy()
            for cid in cid_list:
                if cid == self.ignore_index:
                    continue
                freq_list[cid] = freq_list[cid] + 1
                cid2idxes[cid].append(idx_)
        freq_list = freq_list / (freq_list.sum() + 1e-7)
        class_weight = 1 - freq_list
        class_weight = torch.softmax(torch.from_numpy(class_weight) / self.temperature, dim=0).numpy()

        # saving
        if save_dir is None:
            save_dir = self.data_root
        np.save(osp.join(save_dir, 'class_weight.npy'), class_weight)
        with open(osp.join(save_dir, 'cid2idxes.pkl'), 'wb') as f:
            pickle.dump(cid2idxes, f)
        return class_weight, cid2idxes

    def _rcs(self) -> int:
        """
        Rare class sampling
        Returns:
            idx: an index of the self.data_list
        """
        rand_number, now_sum = np.random.random() * self.weight_sum, 0
        for cid in range(len(self.class_weight)):
            now_sum += self.class_weight[cid]
            if rand_number <= now_sum:
                # print(f'random choose from cid2idxes[{self.METAINFO["classes"][cid]}]')
                return random.choice(self.cid2idxes[cid])

    def __getitem__(self, idx: int) -> dict:
        """Get the idx-th image and data information of dataset after
        ``self.pipeline``, and ``full_init`` will be called if the dataset has
        not been fully initialized.

        During training phase, if ``self.pipeline`` get ``None``,
        ``self._rand_another`` will be called until a valid image is fetched or
         the maximum limit of refetech is reached.

        Args:
            idx (int): The index of self.data_list.

        Returns:
            dict: The idx-th image and data information of dataset after
            ``self.pipeline``.
        """
        # Performing full initialization by calling `__getitem__` will consume
        # extra memory. If a dataset is not fully initialized by setting
        # `lazy_init=True` and then fed into the dataloader. Different workers
        # will simultaneously read and parse the annotation. It will cost more
        # time and memory, although this may work. Therefore, it is recommended
        # to manually call `full_init` before dataset fed into dataloader to
        # ensure all workers use shared RAM from master process.
        if self.test_mode:
            data = self.prepare_data(idx)
            if data is None:
                raise Exception('Test time pipline should not get `None` '
                                'data_sample')
            return data

        for _ in range(self.max_refetch + 1):
            if self.rcs:
                data = self.prepare_data(self._rcs())
                if data is None:
                    continue
            else:
                data = self.prepare_data(idx)
                # Broken images or random augmentations may cause the returned data
                # to be None
                if data is None:
                    idx = self._rand_another()
                    continue
            return data

        raise Exception(f'Cannot find valid image after {self.max_refetch}! '
                        'Please check your image path and pipeline')


@DATASETS.register_module()
class MultiLabelAgricultureVisionDataset(BaseSegDataset):
    METAINFO = dict(
        classes=(
            "background",
            "double_plant",
            "drydown",
            "endrow",
            "nutrient_deficiency",
            "planter_skip",
            "water",
            "waterway",
            "weed_cluster",
        ),
        palette=[
            [0, 0, 0],
            [0, 0, 63],
            [0, 63, 63],
            [0, 63, 0],
            [0, 63, 127],
            [0, 63, 191],
            [0, 63, 255],
            [0, 127, 63],
            [0, 127, 127],
        ],
    )

    def __init__(
            self,
            img_suffix=".tif",
            seg_map_suffix=".tif",
            ignore_index=255,
            rare_class_sampling=False,
            temperature=0.01,
            **kwargs
    ) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            ignore_index=ignore_index,
            **kwargs
        )

        self.class_num = len(self.METAINFO['classes'])
        self.data_list = self.load_data_list()
        print(f'>>>>>>>>>>> loaded {len(self.data_list)} imgs')
        self._fully_initialized = True
        self.eps = 1e-5

        self.rcs = rare_class_sampling
        self.temperature = temperature
        self.class_weight, self.cid2idxes = None, {}
        if self.rcs:
            self.class_weight, self.cid2idxes = self._parse_data(self.data_list)
            self.weight_sum = self.class_weight.sum()

    def _parse_data(self, data_list, save_dir=None):
        """
        parse datalist and get class weights and cid2idxes.
        Args:
            data_list: pathes for images and segmaps, List[dict{}, ...]

        Returns:
            class_weight: Tensor, List[float, ...]
            cid2idxes: dict, dict{int: [dict(), ...], ...}
        """
        # loading exist class_weight and cid2idxes
        path_class_weight = osp.join(self.data_root, 'class_weight.npy')
        path_cid2idxes = osp.join(self.data_root, 'cid2idxes.pkl')
        if os.path.exists(path_class_weight) and os.path.exists(path_cid2idxes):
            logging.log(level=0, msg=f'loading from {path_class_weight}, {path_cid2idxes}.')
            class_weight = np.load(path_class_weight)
            with open(path_cid2idxes, 'rb') as f:
                cid2idxes = pickle.load(f)
            return class_weight, cid2idxes

        class ListQuickAppend:
            def __init__(self, size=65535):
                self.data = [65535] * size
                self.len = 0
            def append(self, v):
                self.data[self.len] = v
                self.len += 1

            def dump2list(self):
                return self.data[: self.len]

        # if the class weight is not pre-inited, parse the labels and init the class weight
        logging.log(level=0, msg=f'parsing train set, getting class weight and cid2idxes.')
        freq_list = np.zeros([self.class_num, ]).astype(np.float32)
        cid2idxes = {cid: ListQuickAppend() for cid in range(self.class_num)}
        for idx_, data_ in tqdm(enumerate(data_list)):
            seg_map_path = data_['seg_map_path']
            seg_map = tifffile.imread(seg_map_path)  # (h, w, c)
            cid_cnt = np.sum(seg_map, axis=(0, 1))
            is_multi_label = np.sum(cid_cnt[1:] > 0) > 1
            for cid_, cnt_ in enumerate(cid_cnt):
                if cid_ == 0 or cid_ == self.ignore_index:
                    continue
                if cnt_ > 0:
                    freq_list[cid_] += 1
                    cid2idxes[cid_].append(idx_)
                    # multi-label
                    if is_multi_label:
                        freq_list[cid_] += 1
                        cid2idxes[cid_].append(idx_)
                    # for endrow
                    if cid_ == 3:
                        freq_list[cid_] += 1
                        cid2idxes[cid_].append(idx_)
            if (idx_ + 1) % 1000 == 0:
                print(freq_list)
                # break
        freq_list[0] = np.sum(freq_list[1:])
        cid2idxes[0] = [idx_ for idx_ in range(len(self.data_list))]
        for cid_ in range(self.class_num - 1):
            cid2idxes[cid_ + 1] = cid2idxes[cid_ + 1].dump2list()

        freq_max = freq_list.max()

        class_weight = [(freq_max / i) if i != 0 else 0 for i in freq_list]
        class_weight = np.array(class_weight, dtype=np.float32)

        # saving
        if save_dir is None:
            save_dir = self.data_root
        np.save(osp.join(save_dir, 'class_weight.npy'), class_weight)
        with open(osp.join(save_dir, 'cid2idxes.pkl'), 'wb') as f:
            pickle.dump(cid2idxes, f)
        return class_weight, cid2idxes

    def _rcs(self) -> int:
        """
        Rare class sampling
        Returns:
            idx: an index of the self.data_list
        """
        rand_number, now_sum = np.random.random() * self.weight_sum, 0
        for cid in range(self.class_num):
            now_sum += self.class_weight[cid]
            if rand_number <= now_sum:
                # print(f'random choose from cid2idxes[{self.METAINFO["classes"][cid]}]')
                return random.choice(self.cid2idxes[cid])

    def __getitem__(self, idx: int) -> dict:
        """Get the idx-th image and data information of dataset after
        ``self.pipeline``, and ``full_init`` will be called if the dataset has
        not been fully initialized.

        During training phase, if ``self.pipeline`` get ``None``,
        ``self._rand_another`` will be called until a valid image is fetched or
         the maximum limit of refetech is reached.

        Args:
            idx (int): The index of self.data_list.

        Returns:
            dict: The idx-th image and data information of dataset after
            ``self.pipeline``.
        """
        # Performing full initialization by calling `__getitem__` will consume
        # extra memory. If a dataset is not fully initialized by setting
        # `lazy_init=True` and then fed into the dataloader. Different workers
        # will simultaneously read and parse the annotation. It will cost more
        # time and memory, although this may work. Therefore, it is recommended
        # to manually call `full_init` before dataset fed into dataloader to
        # ensure all workers use shared RAM from master process.
        if self.test_mode:
            data = self.prepare_data(idx)
            if data is None:
                raise Exception('Test time pipline should not get `None` '
                                'data_sample')
            return data

        for _ in range(self.max_refetch + 1):
            if self.rcs:
                data = self.prepare_data(self._rcs())
                if data is None:
                    continue
            else:
                data = self.prepare_data(idx)
                # Broken images or random augmentations may cause the returned data
                # to be None
                if data is None:
                    idx = self._rand_another()
                    continue
            return data

        raise Exception(f'Cannot find valid image after {self.max_refetch}! '
                        'Please check your image path and pipeline')
