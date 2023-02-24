import torch


def merge_cls_loc_weight(model_weight_old, model_weight_incre):
    # model_weight_old = "/home/zhq/papercode/faster-rcnn-pytorch-master/logs/b19_n1/base/best_epoch_weights.pth"
    mwd = torch.load(model_weight_old)
    name_old = ['head.cls_loc.weight', 'head.cls_loc.bias', 'head.score.weight', 'head.score.bias']
    # (80,2048) (80) (20,2048) (20)

    # model_weight_incre = "/home/zhq/papercode/faster-rcnn-pytorch-master/logs/b19_n1/incre/best_epoch_weights.pth"
    mwi = torch.load(model_weight_incre)
    name_incre = ['head.cls_loc.weight', 'head.cls_loc.bias', 'head.score.weight', 'head.score.bias']
    # (8,2048) (8) (2,2048) (2)

    merge_weight = {}
    merge_weight[name_incre[0]] = torch.cat([mwd[name_incre[0]], mwi[name_incre[0]][4:, :]], dim=0)
    merge_weight[name_incre[1]] = torch.cat([mwd[name_incre[1]], mwi[name_incre[1]][4:]], dim=0)
    merge_weight[name_incre[2]] = torch.cat([mwd[name_incre[2]], mwi[name_incre[2]][1:, :]], dim=0)
    merge_weight[name_incre[3]] = torch.cat([mwd[name_incre[3]], mwi[name_incre[3]][1:]], dim=0)
    for key, value in merge_weight.items():
        mwi[key] = value

    # for i in name_old:
    #     print(mwd[i].shape)
    return mwi


if __name__ == '__main__':
    # model = FasterRCNN(21, anchor_scales=[8, 16, 32], backbone='resnet50', pretrained=False)

    # for n,p in model.rpn.loc.named_parameters():
    #     print(n,p.shape)

    model = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=False)
    print(model)
