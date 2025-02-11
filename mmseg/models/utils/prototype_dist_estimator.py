"""
@Project : mmseg-agri
@File    : 11.py
@IDE     : PyCharm
@Author  : Wang Liu
@Date    : 2025/2/10 上午11:00
@e-mail  : 1183862787@qq.com
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as tnf
import torch.distributed as dist


class DownscaleLabel(nn.Module):

    def __init__(self,
                 scale_factor=32,
                 n_classes=19,
                 ignore_label=255,
                 min_ratio=0.75):
        super().__init__()
        assert scale_factor >= 1
        self.scale_factor = scale_factor
        self.n_classes = n_classes
        self.ignore_label = ignore_label
        self.min_ratio = min_ratio

    def forward(self, label):
        """
        down-scaling annotations
        Args:
            label: annotations, shape=(bs, h, w)
        Returns:
            label: annotations, shape=(bs, 1, h // s, w // s)
        """
        label = label.clone()
        if len(label.shape) == 4:
            label = label.squeeze(dim=1)
        assert len(label.shape) == 3
        bs, orig_h, orig_w = label.shape
        trg_h, trg_w = orig_h // self.scale_factor, orig_w // self.scale_factor
        label[label == self.ignore_label] = self.n_classes
        out = tnf.one_hot(label, num_classes=self.n_classes + 1).permute(0, 3, 1, 2)
        assert list(out.shape) == [bs, self.n_classes + 1, orig_h, orig_w], out.shape
        out = tnf.avg_pool2d(out.float(), kernel_size=self.scale_factor)
        max_ratio, out = torch.max(out, dim=1, keepdim=True)
        out[out == self.n_classes] = self.ignore_label
        out[max_ratio < self.min_ratio] = self.ignore_label
        assert list(out.shape) == [bs, 1, trg_h, trg_w], out.shape
        return out


class Prototypes(nn.Module):
    def __init__(self,
                 num_classes,
                 feat_channels,
                 scale_factor=32,
                 ignore_index=255,
                 momentum=0.999,
                 resume="",
                 debug=False):
        super(Prototypes, self).__init__()
        self.feature_num = feat_channels
        self.class_num = num_classes
        self.ignore_index = ignore_index
        self.momentum = momentum
        self.debug = debug
        self.iter = 0

        self.downer = DownscaleLabel(
            scale_factor=scale_factor,
            n_classes=num_classes,
            ignore_label=ignore_index,
            min_ratio=0.75
        )

        # init prototype
        if resume:
            print("Loading checkpoint from {}".format(resume))
            checkpoint = torch.load(resume, map_location=torch.device('cpu'))
            proto = checkpoint['proto']
        else:
            proto = torch.zeros(self.class_num, self.feature_num)
        # init prototype as a part of the networks
        # self.proto = nn.Parameter(proto)
        self.register_buffer("proto", proto)

    def update(self, features, labels, logits=None):
        if features.ndim == 4:
            features = features.permute(0, 2, 3, 1).reshape(-1, self.feature_num).contiguous()

        # if logits is not None:
        #     logits = tnf.interpolate(logits, size=features.shape[-2:], mode='bilinear', align_corners=False)

        labels = self.downer(labels)
        labels = labels.view(-1).contiguous()

        # remove ignore_index pixels
        mask = (labels != self.ignore_index)
        features = features[mask].detach()
        labels = labels[mask]

        n, k = features.size()
        c = self.class_num
        features = features.view(n, 1, k).expand(n, c, k)
        onehot = torch.zeros(n, c).to(features.device)
        onehot.scatter_(1, labels.view(-1, 1), 1)
        onehot = onehot.view(n, c, 1).expand(n, c, k)
        features = features.mul(onehot)
        counts = onehot.sum(0)
        proto_curr = features.sum(0) / (counts + 1e-5)
        # avoid zero updating
        zero_mask = (counts == 0)  # (c, k)
        proto_curr[zero_mask] = self.proto.data[zero_mask]

        # ema update
        self.proto.data = self.ema(self.proto.data, proto_curr, self.momentum)

        if dist.is_initialized():  # 检查是否初始化了分布式环境
            # 使用 all_reduce 进行同步
            dist.all_reduce(self.proto.data, op=dist.ReduceOp.SUM)  # 先求和
            self.proto.data /= dist.get_world_size()  # 再除以进程数，得到平均值

        if self.debug and (self.iter - 1) % 500 == 0:
            print(self.proto.data[:, :5])

    def ema(self, history, curr, alpha):
        self.iter += 1
        if 1 - (1 / self.iter) < alpha:
            alpha = 1 - (1 / self.iter)
        # if self.debug and (self.iter - 1) % 1000 == 0:
        #     print(alpha)
        return alpha * history + (1 - alpha) * curr

    def save(self, out_path):
        torch.save({'proto': self.proto.data.cpu()}, out_path)

    @staticmethod
    def _cosine_similarity_matrix(tensor_a, tensor_b):
        # 计算余弦相似度矩阵
        norm_a = tnf.normalize(tensor_a, p=2, dim=1)  # 对 tensor_a 做归一化
        norm_b = tnf.normalize(tensor_b, p=2, dim=1)  # 对 tensor_b 做归一化
        sim_matrix = torch.mm(norm_a, norm_b.T)  # 矩阵相乘得到相似度矩阵
        return sim_matrix

    def similarity_cosine(self, feats):
        bs, k, h, w = feats.shape
        feats = feats.detach().permute(0, 2, 3, 1).reshape(-1, k)
        cosine_sim = self._cosine_similarity_matrix(feats, self.proto)
        cosine_sim = cosine_sim.reshape(bs, h, w, self.class_num).permute(0, 3, 1, 2)
        return cosine_sim


if __name__ == '__main__':
    ps = Prototypes(9, 1024, 1, 0,
                    momentum=0.99, debug=True).cuda()
    feats_ = torch.rand((2, 1024, 32, 32)).cuda()
    labels_ = torch.randint(0, 9, (2, 32, 32)).cuda()
    for i in range(200):
        ps.update(feats_, labels_)
        if i == 0:
            feats_ = feats_ * 2
