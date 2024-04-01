import time

import numpy as np
from scipy import ndimage
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from models.modules import *
from models.vit import get_b16_config
import softpool_cuda
from SoftPool import soft_pool2d, SoftPool2d
from models.ScConv import ScConv



class TSTransformer(nn.Module):
    def __init__(self, config, img_size=448, num_classes=200, dataset='cub', smooth_value=0.,
                 loss_alpha=0.4, cam=True, dsm=True, fix=True, update_warm=500,
                 s_perhead=24, total_num=126, assess=False):
        super(TSTransformer, self).__init__()
        self.assess = assess
        self.smooth_value = smooth_value
        self.num_classes = num_classes
        self.loss_alpha = loss_alpha
        self.cam = cam

        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = TSEncoder(config, update_warm, s_perhead, dataset, cam, dsm,
                                   fix, total_num, assess)
        self.head = Linear(config.hidden_size, num_classes)
        self.softmax = Softmax(dim=-1)
        self.SE = SCA_Block(ch_in=3,reduction=3)


    def forward(self, x, labels=None):
        test_mode = False if labels is not None else True
        x = self.SE(x)
        x = self.embeddings(x)
        if self.assess:
            x, xc, assess_list = self.encoder(x, test_mode)
        else:
            x, xc = self.encoder(x, test_mode)

        if self.cam:
            complement_logits = self.head(xc)
            probability = self.softmax(complement_logits)
            weight = self.head.weight
            assist_logit = probability * (weight.sum(-1))
            part_logits = self.head(x) + assist_logit
        else:
            part_logits = self.head(x)

        if self.assess:
            return part_logits, assess_list

        elif test_mode:
            return part_logits

        else:
            if self.smooth_value == 0:
                loss_fct = CrossEntropyLoss()
            else:
                loss_fct = LabelSmoothing(self.smooth_value)

            if self.cam:
                loss_p = loss_fct(part_logits.view(-1, self.num_classes), labels.view(-1))
                loss_c = loss_fct(complement_logits.view(-1, self.num_classes), labels.view(-1))
                contrast_loss = con_loss_new(complement_logits.view(-1, self.num_classes), labels.view(-1))
                alpha = self.loss_alpha
                #loss = (0.9 - alpha) * loss_p + alpha * loss_c + 0.1 * contrast_loss
                #loss = (1 - alpha) * loss_p + alpha * loss_c
                loss = (1 - alpha) * loss_p + alpha * loss_c
            else:
                loss = loss_fct(part_logits.view(-1, self.num_classes), labels.view(-1))
            return part_logits, loss

    def get_eval_data(self):
        return self.encoder.select_num

    def load_from(self, weights):
        with torch.no_grad():
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

            self.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
            self.embeddings.cls_token.copy_(np2th(weights["cls"]))
            # self.encoder.patch_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            # self.encoder.patch_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))
            # self.encoder.clr_encoder.patch_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            # self.encoder.clr_encoder.patch_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.embeddings.position_embeddings.copy_(posemb)
            else:
                ntok_new = posemb_new.size(1)

                posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
                ntok_new -= 1

                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                # print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                posemb_grid = posemb_grid.reshape((1, gs_new * gs_new, -1))
                posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)
                self.embeddings.position_embeddings.copy_(np2th(posemb))

            # for bname, block in self.encoder.named_children():
            # 	for uname, unit in block.named_children():
            # 		if not bname.startswith('key') and not bname.startswith('clr'):
            # 			if uname == '12':
            # 				uname = '11'
            # 			unit.load_from(weights, n_block=uname)
            for bname, block in self.encoder.named_children():
                for uname, unit in block.named_children():
                    # Check if the unit is not Conv2d
                    if not bname.startswith('key') and not bname.startswith('clr') and not isinstance(unit, (nn.Conv2d, nn.Sequential)):
                        if uname == '12':
                            uname = '11'
                        unit.load_from(weights, n_block=uname)


def con_loss_new(features, labels):
    eps = 1e-6

    B, _ = features.shape
    features = F.normalize(features)
    cos_matrix = features.mm(features.t())

    pos_label_matrix = torch.stack([labels == labels[i] for i in range(B)]).float()
    neg_label_matrix = 1 - pos_label_matrix

    neg_label_matrix_new = 1 - pos_label_matrix

    pos_cos_matrix = 1 - cos_matrix
    neg_cos_matrix = 1 + cos_matrix

    margin = 0.3

    sim = (1 + cos_matrix) / 2.0
    scores = 1 - sim

    positive_scores = torch.where(pos_label_matrix == 1.0, scores, scores - scores)
    mask = torch.eye(features.size(0)).cuda()
    positive_scores = torch.where(mask == 1.0, positive_scores - positive_scores, positive_scores)

    positive_scores = torch.sum(positive_scores, dim=1, keepdim=True) / (
                (torch.sum(pos_label_matrix, dim=1, keepdim=True) - 1) + eps)
    positive_scores = torch.repeat_interleave(positive_scores, B, dim=1)

    relative_dis1 = margin + positive_scores - scores
    neg_label_matrix_new[relative_dis1 < 0] = 0
    neg_label_matrix = neg_label_matrix * neg_label_matrix_new

    loss = (pos_cos_matrix * pos_label_matrix).sum() + (neg_cos_matrix * neg_label_matrix).sum()
    loss /= B * B

    return loss
class MS(nn.Module):
    def __init__(self, config, s_perhead=24, fix=True):
        super(MS, self).__init__()
        self.fix = fix
        self.num_heads = config.num_heads
        self.s_perhead = s_perhead
        self.conv = nn.Conv2d(1, 1, 3, 1, 1,dilation=2)

    def forward(self, x, select_num=None, last=False):
        B, patch_num = x.shape[0], x.shape[3] - 1
        select_num = self.s_perhead if select_num is None else select_num
        count = torch.zeros((B, patch_num), dtype=torch.int, device='cuda').half()
        score = x[:, :, 0, 1:]
        _, select = torch.topk(score, self.s_perhead, dim=-1)
        select = select.reshape(B, -1)

        for i, b in enumerate(select):
            count[i, :] += torch.bincount(b, minlength=patch_num)

        if not last:
            count = self.enhace_local(count)
            pass

        patch_value, patch_idx = torch.sort(count, dim=-1, descending=True)
        patch_idx += 1
        return patch_idx[:, :select_num], count

    def enhace_local(self, count):
        B, H = count.shape[0], math.ceil(math.sqrt(count.shape[1]))
        count = count.reshape(B, H, H)
        count = count.type(torch.float32)
        count = self.conv(count.unsqueeze(1)).reshape(B, -1)
        return count

class MLFE(nn.Module):
    def __init__(self, config, mlfe_layer):
        super(MLFE, self).__init__()
        self.mlfe_layer = mlfe_layer
        self.mlfe_norm = LayerNorm(config.hidden_size, eps=1e-6)

    def forward(self, x, cls):
        out = [torch.stack(token) for token in x]
        out = torch.stack(out).squeeze(1)
        out = torch.cat((cls, out), dim=1)
        out, weights = self.clr_layer(out)
        out = self.clr_norm(out)
        return out, weights


class SCA_Block(nn.Module):
    def __init__(self, ch_in, reduction=3):
        super(SCA_Block, self).__init__()
        self.pool = SoftPool2d(kernel_size=(1, 1), stride=(448, 448))
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            #nn.GELU(),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        #y = self.avg_pool(x).view(b, c) # squeeze操作
        y = self.fc(y).view(b, c, 1, 1) # FC获取通道注意力权重，是具有全局信息的
        return x * y.expand_as(x) # 注意力作用每一个通道上



if __name__ == '__main__':
    start = time.time()
    config = get_b16_config()
    # com = clrEncoder(config,)
    # com.to(device='cuda')
    net = TSTransformer(config).cuda()
    # hidden_state = torch.arange(400*768).reshape(2,200,768)/1.0
    x = torch.rand(4, 3, 448, 448, device='cuda')
    y = net(x)
    print(y.shape)
