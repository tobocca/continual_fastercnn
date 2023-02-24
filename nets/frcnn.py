import torch.nn as nn

from nets.classifier import Resnet50RoIHead, VGG16RoIHead
from nets.resnet50 import resnet50
from nets.rpn import RegionProposalNetwork
from nets.vgg16 import decom_vgg16


class FasterRCNN(nn.Module):
    def __init__(self, num_classes,
                 mode="training",
                 feat_stride=16,
                 anchor_scales=[8, 16, 32],
                 ratios=[0.5, 1, 2],
                 backbone='vgg',
                 pretrained=False):
        super(FasterRCNN, self).__init__()
        self.feat_stride = feat_stride
        # ---------------------------------#
        #   一共存在两个主干
        #   vgg和resnet50
        # ---------------------------------#
        if backbone == 'vgg':
            self.extractor, classifier = decom_vgg16(pretrained)
            # ---------------------------------#
            #   构建建议框网络
            # ---------------------------------#
            self.rpn = RegionProposalNetwork(
                512, 512,
                ratios=ratios,
                anchor_scales=anchor_scales,
                feat_stride=self.feat_stride,
                mode=mode
            )
            # ---------------------------------#
            #   构建分类器网络
            # ---------------------------------#
            self.head = VGG16RoIHead(
                n_class=num_classes + 1,
                roi_size=7,
                spatial_scale=1,
                classifier=classifier
            )
        elif backbone == 'resnet50':
            self.extractor, classifier = resnet50(pretrained)
            # ---------------------------------#
            #   构建classifier网络
            # ---------------------------------#
            self.rpn = RegionProposalNetwork(
                1024, 512,
                ratios=ratios,
                anchor_scales=anchor_scales,
                feat_stride=self.feat_stride,
                mode=mode
            )
            # ---------------------------------#
            #   构建classifier网络
            # ---------------------------------#
            self.head = Resnet50RoIHead(
                n_class=num_classes + 1,
                roi_size=14,
                spatial_scale=1,
                classifier=classifier
            )

    def forward(self, x, scale=1., mode="forward"):
        if mode == "forward":
            # ---------------------------------#
            #   计算输入图片的大小
            # ---------------------------------#
            img_size = x.shape[2:]
            # ---------------------------------#
            #   利用主干网络提取特征
            # ---------------------------------#
            base_feature = self.extractor.forward(x)

            # ---------------------------------#
            #   获得建议框
            # ---------------------------------#
            _, _, rois, roi_indices, _ = self.rpn.forward(base_feature, img_size, scale)
            # ---------------------------------------#
            #   获得classifier的分类结果和回归结果
            # ---------------------------------------#
            roi_cls_locs, roi_scores = self.head.forward(base_feature, rois, roi_indices, img_size)
            return roi_cls_locs, roi_scores, rois, roi_indices
        elif mode == "extractor":
            # ---------------------------------#
            #   利用主干网络提取特征
            # ---------------------------------#
            base_feature = self.extractor.forward(x)
            return base_feature
        elif mode == "rpn":
            base_feature, img_size = x
            # ---------------------------------#
            #   获得建议框
            # ---------------------------------#
            rpn_locs, rpn_scores, rois, roi_indices, anchor = self.rpn.forward(base_feature, img_size, scale)
            return rpn_locs, rpn_scores, rois, roi_indices, anchor
        elif mode == "head":
            base_feature, rois, roi_indices, img_size = x
            # ---------------------------------------#
            #   获得classifier的分类结果和回归结果
            # ---------------------------------------#
            roi_cls_locs, roi_scores = self.head.forward(base_feature, rois, roi_indices, img_size)
            return roi_cls_locs, roi_scores

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def freeze_rpn_public(self):
        for p in self.rpn.conv1.parameters():
            p.requires_grad = False

    def freeze_backbone(self):
        for p in self.extractor.parameters():
            p.requires_grad = False

    def freeze_rpn_score(self):
        for p in self.rpn.score.parameters():
            p.requires_grad = False

    def freeze_classifier(self):
        for p in self.head.classifier.parameters():
            p.requires_grad = False


if __name__ == '__main__':
    model = FasterRCNN(20, backbone='resnet50')
    print(model.head)
    # import torch
    # from torchviz import make_dot
    #
    # x = torch.randn(1, 3, 800, 800).requires_grad_(True)  # 定义一个网络的输入值
    # y = model(x)  # 获取网络的预测值
    # MyConvNetVis = make_dot(y, params=dict(list(model.named_parameters()) + [('x', x)]))
    # MyConvNetVis.format = "png"
    # # 指定文件生成的文件夹
    # MyConvNetVis.directory = "./zhq_vis/"
    # # 生成文件
    # MyConvNetVis.view()

    # import hiddenlayer as h
    # vis_graph = h.build_graph(model, torch.zeros([1, 3, 800, 800]))  # 获取绘制图像的对象
    # vis_graph.theme = h.graph.THEMES["blue"].copy()  # 指定主题颜色
    # vis_graph.save("zhq_vis/demo1.png")  # 保存图像的路径
