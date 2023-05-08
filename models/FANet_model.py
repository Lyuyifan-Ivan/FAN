import torch
from torch import nn
from torchcrf import CRF
from transformers.modeling_outputs import TokenClassifierOutput
from .img_extraction_module import Focus, Conv, BottleneckCSP, SPP, Concat
from .spatial_channel_attention import SpatialAttention, ChannelAttention
from transformers import BertModel, XLMRobertaModel
from .cross_attention import FeedForward, EncoderLayer, self_attention
from copy import deepcopy


class ImageModel(nn.Module):
    def __init__(self, max_seq, hidden_size, negative_slope1, negative_slope2):
        super(ImageModel, self).__init__()
        self.max_seq = max_seq
        self.hidden_size = hidden_size
        self.focus = Focus(c1=3, c2=64, k=3, negative_slope=negative_slope2)
        self.cbl_1 = Conv(c1=64, c2=128, k=3, s=2, negative_slope=negative_slope1)
        self.csp_1 = BottleneckCSP(c1=128, c2=128, n=3, negative_slope=negative_slope2)

        self.cbl_2 = Conv(c1=128, c2=256, k=3, s=2, negative_slope=negative_slope1)
        self.csp_2 = BottleneckCSP(c1=256, c2=256, n=6, negative_slope=negative_slope2)  # layer4

        self.cbl_3 = Conv(c1=256, c2=512, k=3, s=2, negative_slope=negative_slope1)
        self.csp_3 = BottleneckCSP(c1=512, c2=512, n=9, negative_slope=negative_slope2)  # layer6

        self.cbl_4 = Conv(c1=512, c2=1024, k=3, s=2, negative_slope=negative_slope1)
        self.csp_4 = BottleneckCSP(c1=1024, c2=1024, n=3, negative_slope=negative_slope2)
        self.ssp = SPP(c1=1024, c2=1024, k=[5, 9, 13], negative_slope=negative_slope2)

        self.spatial_atten = SpatialAttention()
        self.channel_atten = ChannelAttention(in_planes=1024)

        self.cbl_5 = Conv(c1=1024, c2=512, k=1, s=1, negative_slope=negative_slope1)
        self.upsample1 = nn.Upsample(size=None, scale_factor=2, mode='nearest')
        self.concat1 = Concat(1)
        self.csp_5 = BottleneckCSP(c1=512 + 512, c2=512, n=3, shortcut=False, negative_slope=negative_slope2)

        self.cbl_6 = Conv(c1=512, c2=256, k=1, s=1, negative_slope=negative_slope1)
        self.upsample2 = nn.Upsample(size=None, scale_factor=2, mode='nearest')
        self.concat2 = Concat(1)
        self.csp_6 = BottleneckCSP(c1=256 + 256, c2=256, n=3, shortcut=False, negative_slope=negative_slope2)

        self.cbl_7 = Conv(c1=256, c2=256, k=3, s=2, negative_slope=negative_slope1)
        self.concat3 = Concat(1)
        self.csp_7 = BottleneckCSP(c1=256 + 256, c2=512, n=3, shortcut=False, negative_slope=negative_slope2)

        self.cbl_8 = Conv(c1=512, c2=512, k=3, s=2, negative_slope=negative_slope1)
        self.concat4 = Concat(1)
        self.csp_8 = BottleneckCSP(c1=512 + 512, c2=1024, n=3, shortcut=False, negative_slope=negative_slope2)

        self.cbl_final = Conv(c1=1024, c2=self.hidden_size, k=1, s=1, negative_slope=negative_slope1)
        self.csp_final = BottleneckCSP(c1=self.hidden_size, c2=self.hidden_size, n=3, shortcut=True, negative_slope=negative_slope2)

    def forward(self, x):
        final_layer = self.get_img_feature(x)
        return final_layer

    def get_img_feature(self, x):
        layer4 = self.csp_2(self.cbl_2(self.csp_1(self.cbl_1(self.focus(x)))))
        layer6 = self.csp_3(self.cbl_3(layer4))
        layer9 = self.ssp(self.csp_4(self.cbl_4(layer6)))

        layer10 = self.cbl_5(layer9)
        layer13 = self.csp_5(self.concat1([self.upsample1(layer10), layer6]))

        layer14 = self.cbl_6(layer13)
        layer17 = self.csp_6(self.concat2([self.upsample2(layer14), layer4]))

        layer20 = self.csp_7(self.concat3([self.cbl_7(layer17), layer14]))

        layer23 = self.csp_8(self.concat4([self.cbl_8(layer20), layer10]))

        layer23 = layer23.mul(self.channel_atten(layer23))
        layer23 = layer23.mul(self.spatial_atten(layer23))

        final_layer = self.csp_final(self.cbl_final(layer23))

        return final_layer


class TI2FANetModel(nn.Module):
    def __init__(self, label_list, args):
        super(TI2FANetModel, self).__init__()
        self.args = args
        self.bert = BertModel.from_pretrained(args.bert_name)
        self.bert_config = self.bert.config

        self.image_model = ImageModel(args.max_seq, self.bert_config.hidden_size, args.negative_slope1, args.negative_slope2)
        
        self.num_labels = len(label_list)  # 13

        self.dropout = nn.Dropout(args.dropout)

        self.final_dropout = nn.Dropout(args.final_dropout)

        self.feed_forward = FeedForward(self.bert_config.hidden_size, self.bert_config.hidden_size)
        self.Cross_atten = EncoderLayer(d_model=self.bert_config.hidden_size, attn=deepcopy(self_attention), feed_forward=deepcopy(self.feed_forward))

        self.classifier = nn.Linear(self.bert_config.hidden_size, self.num_labels)

        self.crf = CRF(self.num_labels, batch_first=True)
        
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, images=None):
        img_output = self.get_visual_convert(images)

        bert_output = self.bert(input_ids=input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids,
                                return_dict=True)
        sequence_output = bert_output['last_hidden_state']  # bsz, len, hidden

        sequence_output = self.dropout(sequence_output)  # bsz, len, hidden

        cross_output = self.Cross_atten(sequence_output, img_output)

        emissions = self.classifier(cross_output)  # bsz, len, num_labels

        emissions = self.final_dropout(emissions)

        logits = self.crf.decode(emissions, attention_mask.byte())
        loss = None
        if labels is not None:
            loss = -1 * self.crf(emissions, labels, mask=attention_mask.byte(), reduction='mean')
        return TokenClassifierOutput(
            loss=loss,
            logits=logits
        )

    def get_visual_convert(self, images):
        bsz = images.size(0)
        img_feat = self.image_model(images)  # [bsz, hidden_size, 7, 7]
        return img_feat.view(bsz, -1, self.bert_config.hidden_size)

