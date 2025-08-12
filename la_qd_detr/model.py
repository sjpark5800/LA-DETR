# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn

from la_qd_detr.span_utils import generalized_temporal_iou, span_cxw_to_xx

from la_qd_detr.matcher import build_matcher
from la_qd_detr.transformer import build_transformer
from la_qd_detr.position_encoding import build_position_encoding
from la_qd_detr.misc import accuracy
import numpy as np
from typing import List

def inverse_sigmoid(x, eps=1e-3):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)


def focal_loss(inputs, targets, weight, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    ce_loss = F.cross_entropy(inputs, targets, weight, reduction='none')
    pt = torch.exp(-ce_loss)
    loss = (alpha * (1 - pt) ** gamma * ce_loss)
    return loss


class MoeLayer(nn.Module):
    def __init__(self, experts: List[nn.Module], num_queries=10):
        super().__init__()
        assert len(experts) > 0
        self.experts = nn.ModuleList(experts)
        self.num_queries = num_queries

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        results = []
        for i, expert in enumerate(self.experts):
            start, end = self.num_queries * i , self.num_queries * (i + 1)
            results.append(expert(inputs[:, :, start : end, :]))
        return torch.cat(results, dim=-2)

class QDDETR(nn.Module):
    """ QD DETR. """

    def __init__(self, transformer, position_embed, txt_position_embed, txt_dim, vid_dim,
                 num_queries, input_dropout, aux_loss=False,
                 contrastive_align_loss=False, contrastive_hdim=64,
                 max_v_l=75, span_loss_type="l1", use_txt_pos=False, n_input_proj=2, aud_dim=0, 
                 m_classes=None, cls_both=False, score_fg=False, class_anchor=False, class_moe=False, span_moe=False,
                 length_query=None,):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture. See transformer.py
            position_embed: torch module of the position_embedding, See position_encoding.py
            txt_position_embed: position_embedding for text
            txt_dim: int, text query input dimension
            vid_dim: int, video feature input dimension
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         QD-DETR can detect in a single video.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            contrastive_align_loss: If true, perform span - tokens contrastive learning
            contrastive_hdim: dimension used for projecting the embeddings before computing contrastive loss
            max_v_l: int, maximum #clips in videos
            span_loss_type: str, one of [l1, ce]
                l1: (center-x, width) regression.
                ce: (st_idx, ed_idx) classification.
            # foreground_thd: float, intersection over prediction >= foreground_thd: labeled as foreground
            # background_thd: float, intersection over prediction <= background_thd: labeled background
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.position_embed = position_embed
        self.txt_position_embed = txt_position_embed
        hidden_dim = transformer.d_model
        self.span_loss_type = span_loss_type
        self.max_v_l = max_v_l

        self.m_classes=m_classes
        self.cls_both=cls_both
        self.score_fg=score_fg

        span_pred_dim = 2 if span_loss_type == "l1" else max_v_l * 2
        if self.m_classes is None:
            self.span_embed = MLP(hidden_dim, hidden_dim, span_pred_dim, 3)
            self.class_embed = nn.Linear(hidden_dim, 2)  # 0: background, 1: foreground
            self.num_patterns = 1
        else:
            self.m_vals = [float(v) for v in m_classes[1:-1].split(',')]
            if class_anchor:
                self.num_patterns = len(self.m_vals)
            else:
                self.num_patterns = 1

            if class_moe:
                self.class_embed = MoeLayer(
                    experts=[nn.Linear(hidden_dim, 2) for _ in range(len(self.m_vals))],
                    num_queries=num_queries
                )       
            else:
                self.class_embed = nn.Linear(hidden_dim, 2)  # 0: background, 1: foreground

            if span_moe:
                self.span_embed = MoeLayer(
                    experts=[MLP(hidden_dim, hidden_dim, span_pred_dim, 3) for _ in range(len(self.m_vals))],
                    num_queries=num_queries
                )
            else:
                self.span_embed = MLP(hidden_dim, hidden_dim, span_pred_dim, 3)
            
                
        self.use_txt_pos = use_txt_pos
        self.n_input_proj = n_input_proj
        # self.foreground_thd = foreground_thd
        # self.background_thd = background_thd
        # if self.m_classes is not None

        if length_query is None:
            self.query_embed = nn.Embedding(num_queries, 2*self.num_patterns)
        else:
            self.len_query_num = [int(lq) for lq in length_query[1:-1].split(',')]
            self.num_queries = sum(self.len_query_num)
            self.query_embed = nn.Embedding(self.num_queries, 2)

        relu_args = [True] * 3
        relu_args[n_input_proj-1] = False
        self.input_txt_proj = nn.Sequential(*[
            LinearLayer(txt_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[0]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[1]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[2])
        ][:n_input_proj])
        self.input_vid_proj = nn.Sequential(*[
            LinearLayer(vid_dim + aud_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[0]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[1]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[2])
        ][:n_input_proj])
        self.contrastive_align_loss = contrastive_align_loss
        if contrastive_align_loss:
            self.contrastive_align_projection_query = nn.Linear(hidden_dim, contrastive_hdim)
            self.contrastive_align_projection_txt = nn.Linear(hidden_dim, contrastive_hdim)
            self.contrastive_align_projection_vid = nn.Linear(hidden_dim, contrastive_hdim)

        self.saliency_proj1 = nn.Linear(hidden_dim, hidden_dim)
        self.saliency_proj2 = nn.Linear(hidden_dim, hidden_dim)
        self.aux_loss = aux_loss

        self.hidden_dim = hidden_dim
        self.global_rep_token = torch.nn.Parameter(torch.randn(hidden_dim))
        self.global_rep_pos = torch.nn.Parameter(torch.randn(hidden_dim))
        
        

    def forward(self, src_txt, src_txt_mask, src_vid, src_vid_mask, src_aud=None, src_aud_mask=None):
        """The forward expects two tensors:
               - src_txt: [batch_size, L_txt, D_txt]
               - src_txt_mask: [batch_size, L_txt], containing 0 on padded pixels,
                    will convert to 1 as padding later for transformer
               - src_vid: [batch_size, L_vid, D_vid]
               - src_vid_mask: [batch_size, L_vid], containing 0 on padded pixels,
                    will convert to 1 as padding later for transformer

            It returns a dict with the following elements:
               - "pred_spans": The normalized boxes coordinates for all queries, represented as
                               (center_x, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if src_aud is not None:
            src_vid = torch.cat([src_vid, src_aud], dim=2)
            
        src_vid = self.input_vid_proj(src_vid)
        src_txt = self.input_txt_proj(src_txt)
        src = torch.cat([src_vid, src_txt], dim=1)  # (bsz, L_vid+L_txt, d)
        mask = torch.cat([src_vid_mask, src_txt_mask], dim=1).bool()  # (bsz, L_vid+L_txt)
        # TODO should we remove or use different positional embeddings to the src_txt?
        pos_vid = self.position_embed(src_vid, src_vid_mask)  # (bsz, L_vid, d)
        pos_txt = self.txt_position_embed(src_txt) if self.use_txt_pos else torch.zeros_like(src_txt)  # (bsz, L_txt, d)
        # pos_txt = torch.zeros_like(src_txt)
        # pad zeros for txt positions
        pos = torch.cat([pos_vid, pos_txt], dim=1)
        # (#layers, bsz, #queries, d), (bsz, L_vid+L_txt, d)

        # for global token
        mask_ = torch.tensor([[True]]).to(mask.device).repeat(mask.shape[0], 1)
        mask = torch.cat([mask_, mask], dim=1)
        src_ = self.global_rep_token.reshape([1, 1, self.hidden_dim]).repeat(src.shape[0], 1, 1)
        src = torch.cat([src_, src], dim=1)
        pos_ = self.global_rep_pos.reshape([1, 1, self.hidden_dim]).repeat(pos.shape[0], 1, 1)
        pos = torch.cat([pos_, pos], dim=1)

        video_length = src_vid.shape[1]
        
        hs, reference, memory, memory_global = self.transformer(src, ~mask, self.query_embed.weight, pos, video_length=video_length)
        outputs_class = self.class_embed(hs)  # (#layers, batch_size, #queries, #classes)
        reference_before_sigmoid = inverse_sigmoid(reference)
        tmp = self.span_embed(hs)
        outputs_coord = tmp + reference_before_sigmoid
        if self.span_loss_type == "l1":
            outputs_coord = outputs_coord.sigmoid()
            
        out = {'pred_logits': outputs_class[-1], 'pred_spans': outputs_coord[-1]}
        if self.m_classes is not None and self.cls_both:
            outputs_aux_class = self.aux_class_embed(hs)
            out['aux_pred_logits'] = outputs_aux_class[-1]

        txt_mem = memory[:, src_vid.shape[1]:]  # (bsz, L_txt, d)
        vid_mem = memory[:, :src_vid.shape[1]]  # (bsz, L_vid, d)
        if self.contrastive_align_loss:
            proj_queries = F.normalize(self.contrastive_align_projection_query(hs), p=2, dim=-1)
            proj_txt_mem = F.normalize(self.contrastive_align_projection_txt(txt_mem), p=2, dim=-1)
            proj_vid_mem = F.normalize(self.contrastive_align_projection_vid(vid_mem), p=2, dim=-1)
            out.update(dict(
                proj_queries=proj_queries[-1],
                proj_txt_mem=proj_txt_mem,
                proj_vid_mem=proj_vid_mem
            ))
            
            
        # !!! this is code for test
        if src_txt.shape[1] == 0:
            print("There is zero text query. You should change codes properly")
            exit(-1)

        ### Neg Pairs ###
        src_txt_neg = torch.cat([src_txt[1:], src_txt[0:1]], dim=0)
        src_txt_mask_neg = torch.cat([src_txt_mask[1:], src_txt_mask[0:1]], dim=0)
        src_neg = torch.cat([src_vid, src_txt_neg], dim=1)
        mask_neg = torch.cat([src_vid_mask, src_txt_mask_neg], dim=1).bool()

        mask_neg = torch.cat([mask_, mask_neg], dim=1)
        src_neg = torch.cat([src_, src_neg], dim=1)
        pos_neg = pos.clone()  # since it does not use actual content

        _, _, memory_neg, memory_global_neg = self.transformer(src_neg, ~mask_neg, self.query_embed.weight, pos_neg, video_length=video_length)
        vid_mem_neg = memory_neg[:, :src_vid.shape[1]]


        out["saliency_scores"] = (torch.sum(self.saliency_proj1(vid_mem) * self.saliency_proj2(memory_global).unsqueeze(1), dim=-1) / np.sqrt(self.hidden_dim))

        out["saliency_scores_neg"] = (torch.sum(self.saliency_proj1(vid_mem_neg) * self.saliency_proj2(memory_global_neg).unsqueeze(1), dim=-1) / np.sqrt(self.hidden_dim))

        # print(src_vid_mask.shape, src_vid.shape, vid_mem_neg.shape, vid_mem.shape)
        out["video_mask"] = src_vid_mask
        if self.aux_loss:
            # assert proj_queries and proj_txt_mem
            if not self.cls_both:
                out['aux_outputs'] = [
                    {'pred_logits': a, 'pred_spans': b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]
            else:
                out['aux_outputs'] = [
                    {'pred_logits': a, 'aux_pred_logits' : b, 'pred_spans': c} for a, b, c in zip(outputs_class[:-1], outputs_aux_class[:-1], outputs_coord[:-1])]
            if self.contrastive_align_loss:
                assert proj_queries is not None
                for idx, d in enumerate(proj_queries[:-1]):
                    out['aux_outputs'][idx].update(dict(proj_queries=d, proj_txt_mem=proj_txt_mem))
        return out

    # @torch.jit.unused
    # def _set_aux_loss(self, outputs_class, outputs_coord):
    #     # this is a workaround to make torchscript happy, as torchscript
    #     # doesn't support dictionary with non-homogeneous values, such
    #     # as a dict having both a Tensor and a list.
    #     return [{'pred_logits': a, 'pred_spans': b}
    #             for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, matcher, weight_dict, eos_coef, losses, temperature, span_loss_type, max_v_l,
                 saliency_margin=1, use_matcher=True, m_classes=None, cls_both=False, score_fg=False,
                 label_loss_type='ce', focal_alpha=0.25, focal_gamma=2.0, 
                 aux_label_loss_type='ce', aux_focal_alpha=0.25, aux_focal_gamma=2.0,
                 length_span_weight=False, length_giou_weight=False):
        """ Create the criterion.
        Parameters:
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            temperature: float, temperature for NCE loss
            span_loss_type: str, [l1, ce]
            max_v_l: int,
            saliency_margin: float
        """
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.eos_coef = eos_coef
        self.temperature = temperature
        self.span_loss_type = span_loss_type
        self.max_v_l = max_v_l
        self.saliency_margin = saliency_margin

        self.m_classes = m_classes
        self.cls_both=cls_both
        self.score_fg=score_fg
        
        # foreground and background classification
        self.foreground_label = 0
        self.background_label = 1

        empty_weight = torch.ones(2)
        empty_weight[-1] = self.eos_coef  # lower weight for background (index 1, foreground index 0)
        self.register_buffer('empty_weight', empty_weight)
        if m_classes is not None: 
            self.num_classes = len(m_classes[1:-1].split(','))
            if self.cls_both:
                aux_empty_weight = torch.ones(self.num_classes+ 1)
                aux_empty_weight[-1] = self.eos_coef  # lower weight for background (index 1, foreground index 0)
                self.register_buffer('aux_empty_weight', aux_empty_weight) 
            
        # for tvsum,
        self.use_matcher = use_matcher

        self.label_loss_type = label_loss_type
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

        self.aux_label_loss_type = aux_label_loss_type
        self.aux_focal_alpha = aux_focal_alpha
        self.aux_focal_gamma = aux_focal_gamma

        self.length_span_weight = length_span_weight
        self.length_giou_weight = length_giou_weight


        if length_giou_weight or length_span_weight:

            # 새로운 정규분포 생성 (조절 가능한 파라미터로 범위 및 감쇠 조정)
            x_values_custom = np.linspace(0, 150, 300)

            # 평균 및 표준편차 설정 (중심: 0, 범위 끝: 150에서 0.7이 되도록 조정)
            mean_custom = 0
            std_dev_custom = 30  # 감쇠를 조절하는 표준편차 (이 값을 조정하여 곡선의 폭을 조절할 수 있음)

            peak_value = 2
            end_value = 1

            # 스케일링 인자를 계산하여 0에서의 값을 1로 만들고, 150에서의 값을 0.7로 유지
            scale_factor = (peak_value - end_value) / (np.exp(-0.5 * ((x_values_custom - mean_custom) / std_dev_custom) ** 2).max())

            # 조정된 정규분포 계산
            pdf_values_custom = scale_factor * np.exp(-0.5 * ((x_values_custom - mean_custom) / std_dev_custom) ** 2) + end_value

            self.weight_dist = torch.tensor(pdf_values_custom, device='cuda')


    def loss_spans(self, outputs, targets, indices):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "spans" containing a tensor of dim [nb_tgt_spans, 2]
           The target spans are expected in format (center_x, w), normalized by the image size.
        """
        assert 'pred_spans' in outputs
        targets_span = targets["span_labels"]
        idx = self._get_src_permutation_idx(indices)
        src_spans = outputs['pred_spans'][idx]  # (#spans, max_v_l * 2)
        tgt_spans = torch.cat([t['spans'][i] for t, (_, i) in zip(targets_span, indices)], dim=0)  # (#spans, 2)
        if self.span_loss_type == "l1":
            loss_span = F.l1_loss(src_spans, tgt_spans, reduction='none')
            loss_giou = 1 - torch.diag(generalized_temporal_iou(span_cxw_to_xx(src_spans), span_cxw_to_xx(tgt_spans)))
        else:  # ce
            n_spans = src_spans.shape[0]
            src_spans = src_spans.view(n_spans, 2, self.max_v_l).transpose(1, 2)
            loss_span = F.cross_entropy(src_spans, tgt_spans, reduction='none')

            # giou
            # src_span_indices = src_spans.max(1)[1]  # (#spans, 2)
            # src_span_indices[:, 1] += 1  # ed non-inclusive [st, ed)
            #
            # tgt_span_indices = tgt_spans
            # tgt_span_indices[:, 1] += 1
            # loss_giou = 1 - torch.diag(generalized_temporal_iou(src_span_indices, tgt_span_indices))
            loss_giou = loss_span.new_zeros([1])

        if self.length_giou_weight or self.length_span_weight:
            target_length = targets['moment_length']
            tgt_length_weight = torch.cat([self.weight_dist[t['m_len'][i]] for t, (_, i) in zip(target_length, indices)], dim=0)  # (#spans, 2)

        if self.length_span_weight:
            loss_span = loss_span.mean(dim=1) * tgt_length_weight

        if self.length_giou_weight:
            loss_giou = loss_giou * tgt_length_weight

        losses = {}
        losses['loss_span'] = loss_span.mean()
        losses['loss_giou'] = loss_giou.mean()

        if 'loss_moment_class' in targets:
            target_cls = targets['loss_moment_class']
            tgt_cls = torch.cat([t['m_cls'][i] for t, (_, i) in zip(target_cls, indices)], dim=0)  # (#spans, 2)

            short_idx = (tgt_cls == 0).nonzero(as_tuple=True)[0]
            medium_idx = (tgt_cls == 1).nonzero(as_tuple=True)[0]
            long_idx = (tgt_cls == 2).nonzero(as_tuple=True)[0]
            # verlong_idx = (tgt_cls == 3).nonzero(as_tuple=True)[0]

            losses['0_num'] = len(short_idx)
            losses['1_num'] = len(medium_idx)
            losses['2_num'] = len(long_idx)
            # losses['3_num'] = len(verlong_idx)

            losses['loss_0_giou_sum'] = loss_giou[short_idx].sum()
            losses['loss_1_giou_sum'] = loss_giou[medium_idx].sum()
            losses['loss_2_giou_sum'] = loss_giou[long_idx].sum()
            # losses['loss_3_giou_sum'] = loss_giou[verlong_idx].sum()
            losses['loss_full_giou_sum'] = loss_giou.sum()

            losses['loss_0_giou_mean'] = loss_giou[short_idx].mean()
            losses['loss_1_giou_mean'] = loss_giou[medium_idx].mean()
            losses['loss_2_giou_mean'] = loss_giou[long_idx].mean()
            # losses['loss_3_giou_mean'] = loss_giou[verlong_idx].mean()
            losses['loss_full_giou_mean'] = loss_giou.mean()

            losses['loss_0_span_sum'] = loss_span[short_idx].sum()
            losses['loss_1_span_sum'] = loss_span[medium_idx].sum()
            losses['loss_2_span_sum'] = loss_span[long_idx].sum()
            # losses['loss_3_spans_sum'] = loss_span[verlong_idx].sum()
            losses['loss_full_span_sum'] = loss_span.sum()

            losses['loss_0_span_mean'] = loss_span[short_idx].mean()
            losses['loss_1_span_mean'] = loss_span[medium_idx].mean()
            losses['loss_2_span_mean'] = loss_span[long_idx].mean()
            # losses['loss_3_spans_mean'] = loss_span[verlong_idx].mean()
            losses['loss_full_span_mean'] = loss_span.mean()

        return losses

    def loss_labels(self, outputs, targets, indices, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        # TODO add foreground and background classifier.  use all non-matched as background.
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']  # (batch_size, #queries, #classes=2)
        # idx is a tuple of two 1D tensors (batch_idx, src_idx), of the same length == #objects in batch
        idx = self._get_src_permutation_idx(indices)

        target_classes = torch.full(src_logits.shape[:2], self.background_label,
                                    dtype=torch.int64, device=src_logits.device)  # (batch_size, #queries)
        target_classes[idx] = self.foreground_label
        
        if self.m_classes is not None and self.cls_both:
            aux_src_logits = outputs['aux_pred_logits']
            aux_target_classes = torch.full(aux_src_logits.shape[:2], self.num_classes,
                                        dtype=torch.int64, device=src_logits.device)  # (batch_size, #queries)

            aux_target_classes_o = torch.cat([t["m_cls"][J] for t, (_, J) in zip(targets['moment_class'], indices)])
            aux_target_classes[idx] = aux_target_classes_o
   
        if self.label_loss_type == "ce":
            loss = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight, reduction="none")
        else:
            loss = focal_loss(src_logits.transpose(1, 2), target_classes, self.empty_weight, self.focal_alpha, self.focal_gamma)

        if self.m_classes is not None and self.cls_both:
            if self.aux_label_loss_type == "ce":
                loss += F.cross_entropy(aux_src_logits.transpose(1, 2), aux_target_classes, self.aux_empty_weight, reduction="none")
            else:
                loss += focal_loss(aux_src_logits.transpose(1, 2), aux_target_classes, self.aux_empty_weight, self.aux_focal_alpha, self.aux_focal_gamma)

        losses = {'loss_label': loss.mean()}

        if 'loss_moment_class' in targets:
            target_cls = targets['loss_moment_class']
            tgt_cls = torch.cat([t['m_cls'][i] for t, (_, i) in zip(target_cls, indices)], dim=0)  # (#spans, 2)

            short_idx = (tgt_cls == 0).nonzero(as_tuple=True)[0]
            medium_idx = (tgt_cls == 1).nonzero(as_tuple=True)[0]
            long_idx = (tgt_cls == 2).nonzero(as_tuple=True)[0]
            # verlong_idx = (tgt_cls == 3).nonzero(as_tuple=True)[0]

            losses['loss_0_fg_label_sum'] = loss[idx][short_idx].sum()
            losses['loss_1_fg_label_sum'] = loss[idx][medium_idx].sum()
            losses['loss_2_fg_label_sum'] = loss[idx][long_idx].sum()
            # losses['loss_3_label_sum'] = loss[verlong_idx].sum()
            losses['loss_full_fg_label_sum'] = loss[idx].sum()
            losses['loss_full_all_label_sum'] = loss.sum()

            losses['loss_0_fg_label_mean'] = loss[idx][short_idx].mean()
            losses['loss_1_fg_label_mean'] = loss[idx][medium_idx].mean()
            losses['loss_2_fg_label_mean'] = loss[idx][long_idx].mean()
            # losses['loss_3_label_mean'] = loss[verlong_idx].mean()
            losses['loss_full_fg_label_mean'] = loss[idx].mean()
            losses['loss_full_all_label_mean'] = loss.mean()

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], self.foreground_label)[0]

            if 'loss_moment_class' in targets:
                if len(short_idx)>0 :
                    losses['class_0_acc'] = accuracy(src_logits[idx][short_idx], self.foreground_label)[0]
                if len(medium_idx)>0 :
                    losses['class_1_acc'] = accuracy(src_logits[idx][medium_idx], self.foreground_label)[0]
                if len(long_idx)>0 :
                    losses['class_2_acc'] = accuracy(src_logits[idx][long_idx], self.foreground_label)[0]
                # losses['class_4_acc'] = accuracy(src_logits[idx][verlong_idx], self.foreground_label)[0] if len(verlong_idx)>0 else -1
                losses['class_full_acc'] = accuracy(src_logits[idx], self.foreground_label)[0]

            if self.m_classes is not None and self.cls_both:
                losses['aux_class_error'] = 100 - accuracy(aux_src_logits[idx], self.foreground_label)[0]
        return losses

    def loss_saliency(self, outputs, targets, indices, log=True):
        """higher scores for positive clips"""
        if "saliency_pos_labels" not in targets:
            return {"loss_saliency": 0}

        vid_token_mask = outputs["video_mask"]

        # Neg pair loss
        saliency_scores_neg = outputs["saliency_scores_neg"].clone()  # (N, L)
        # loss_neg_pair = torch.sigmoid(saliency_scores_neg).mean()
        
        loss_neg_pair = (- torch.log(1. - torch.sigmoid(saliency_scores_neg)) * vid_token_mask).sum(dim=1).mean()

        saliency_scores = outputs["saliency_scores"].clone()  # (N, L)
        saliency_contrast_label = targets["saliency_all_labels"]

        saliency_scores = torch.cat([saliency_scores, saliency_scores_neg], dim=1)
        saliency_contrast_label = torch.cat([saliency_contrast_label, torch.zeros_like(saliency_contrast_label)], dim=1)

        vid_token_mask = vid_token_mask.repeat([1, 2])
        saliency_scores = vid_token_mask * saliency_scores + (1. - vid_token_mask) * -1e+3

        tau = 0.5
        loss_rank_contrastive = 0.

        # for rand_idx in range(1, 13, 3):
        #     # 1, 4, 7, 10 --> 5 stages
        for rand_idx in range(1, 12):
            drop_mask = ~(saliency_contrast_label > 100)  # no drop
            pos_mask = (saliency_contrast_label >= rand_idx)  # positive when equal or higher than rand_idx

            if torch.sum(pos_mask) == 0:  # no positive sample
                continue
            else:
                batch_drop_mask = torch.sum(pos_mask, dim=1) > 0  # negative sample indicator

            # drop higher ranks
            cur_saliency_scores = saliency_scores * drop_mask / tau + ~drop_mask * -1e+3

            # numerical stability
            logits = cur_saliency_scores - torch.max(cur_saliency_scores, dim=1, keepdim=True)[0]

            # softmax
            exp_logits = torch.exp(logits)
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

            mean_log_prob_pos = (pos_mask * log_prob * vid_token_mask).sum(1) / (pos_mask.sum(1) + 1e-6)

            loss = - mean_log_prob_pos * batch_drop_mask

            loss_rank_contrastive = loss_rank_contrastive + loss.mean()

        loss_rank_contrastive = loss_rank_contrastive / 12

        saliency_scores = outputs["saliency_scores"]  # (N, L)
        pos_indices = targets["saliency_pos_labels"]  # (N, #pairs)
        neg_indices = targets["saliency_neg_labels"]  # (N, #pairs)
        num_pairs = pos_indices.shape[1]  # typically 2 or 4
        batch_indices = torch.arange(len(saliency_scores)).to(saliency_scores.device)
        pos_scores = torch.stack(
            [saliency_scores[batch_indices, pos_indices[:, col_idx]] for col_idx in range(num_pairs)], dim=1)
        neg_scores = torch.stack(
            [saliency_scores[batch_indices, neg_indices[:, col_idx]] for col_idx in range(num_pairs)], dim=1)
        loss_saliency = torch.clamp(self.saliency_margin + neg_scores - pos_scores, min=0).sum() \
                        / (len(pos_scores) * num_pairs) * 2  # * 2 to keep the loss the same scale

        # print(loss_saliency, loss_rank_contrastive)
        # loss_saliency = loss_saliency + loss_rank_contrastive
        loss_saliency = loss_saliency + loss_rank_contrastive + loss_neg_pair
        # loss_saliency = loss_rank_contrastive
        return {"loss_saliency": loss_saliency}

    def loss_contrastive_align(self, outputs, targets, indices, log=True):
        """encourage higher scores between matched query span and input text"""
        normalized_text_embed = outputs["proj_txt_mem"]  # (bsz, #tokens, d)  text tokens
        normalized_img_embed = outputs["proj_queries"]  # (bsz, #queries, d)
        logits = torch.einsum(
            "bmd,bnd->bmn", normalized_img_embed, normalized_text_embed)  # (bsz, #queries, #tokens)
        logits = logits.sum(2) / self.temperature  # (bsz, #queries)
        idx = self._get_src_permutation_idx(indices)
        positive_map = torch.zeros_like(logits, dtype=torch.bool)
        positive_map[idx] = True
        positive_logits = logits.masked_fill(~positive_map, 0)

        pos_term = positive_logits.sum(1)  # (bsz, )
        num_pos = positive_map.sum(1)  # (bsz, )
        neg_term = logits.logsumexp(1)  # (bsz, )
        loss_nce = - pos_term / num_pos + neg_term  # (bsz, )
        losses = {"loss_contrastive_align": loss_nce.mean()}
        return losses

    def loss_contrastive_align_vid_txt(self, outputs, targets, indices, log=True):
        """encourage higher scores between matched query span and input text"""
        # TODO (1)  align vid_mem and txt_mem;
        # TODO (2) change L1 loss as CE loss on 75 labels, similar to soft token prediction in MDETR
        normalized_text_embed = outputs["proj_txt_mem"]  # (bsz, #tokens, d)  text tokens
        normalized_img_embed = outputs["proj_queries"]  # (bsz, #queries, d)
        logits = torch.einsum(
            "bmd,bnd->bmn", normalized_img_embed, normalized_text_embed)  # (bsz, #queries, #tokens)
        logits = logits.sum(2) / self.temperature  # (bsz, #queries)
        idx = self._get_src_permutation_idx(indices)
        positive_map = torch.zeros_like(logits, dtype=torch.bool)
        positive_map[idx] = True
        positive_logits = logits.masked_fill(~positive_map, 0)

        pos_term = positive_logits.sum(1)  # (bsz, )
        num_pos = positive_map.sum(1)  # (bsz, )
        neg_term = logits.logsumexp(1)  # (bsz, )
        loss_nce = - pos_term / num_pos + neg_term  # (bsz, )
        losses = {"loss_contrastive_align": loss_nce.mean()}
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx  # two 1D tensors of the same length

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, **kwargs):
        loss_map = {
            "spans": self.loss_spans,
            "labels": self.loss_labels,
            "contrastive_align": self.loss_contrastive_align,
            "saliency": self.loss_saliency,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        # list(tuples), each tuple is (pred_span_indices, tgt_span_indices)

        # only for HL, do not use matcher
        if self.use_matcher:
            indices = self.matcher(outputs_without_aux, targets)
            losses_target = self.losses
        else:
            indices = None
            losses_target = ["saliency"]

        # Compute all the requested losses
        losses = {}
        # for loss in self.losses:
        for loss in losses_target:
            losses.update(self.get_loss(loss, outputs, targets, indices))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                # indices = self.matcher(aux_outputs, targets)
                if self.use_matcher:
                    indices = self.matcher(aux_outputs, targets)
                    losses_target = self.losses
                else:
                    indices = None
                    losses_target = ["saliency"]    
                # for loss in self.losses:
                for loss in losses_target:
                    if "saliency" == loss:  # skip as it is only in the top layer
                        continue
                    kwargs = {}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class LinearLayer(nn.Module):
    """linear layer configurable with layer normalization, dropout, ReLU."""

    def __init__(self, in_hsz, out_hsz, layer_norm=True, dropout=0.1, relu=True):
        super(LinearLayer, self).__init__()
        self.relu = relu
        self.layer_norm = layer_norm
        if layer_norm:
            self.LayerNorm = nn.LayerNorm(in_hsz)
        layers = [
            nn.Dropout(dropout),
            nn.Linear(in_hsz, out_hsz)
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """(N, L, D)"""
        if self.layer_norm:
            x = self.LayerNorm(x)
        x = self.net(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x  # (N, L, D)


def build_model(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/qd_detr/issues/108#issuecomment-650269223
    device = torch.device(args.device)

    transformer = build_transformer(args)
    position_embedding, txt_position_embedding = build_position_encoding(args)

    if args.a_feat_dir is None:
        model = QDDETR(
            transformer,
            position_embedding,
            txt_position_embedding,
            txt_dim=args.t_feat_dim,
            vid_dim=args.v_feat_dim,
            num_queries=args.num_queries,
            input_dropout=args.input_dropout,
            aux_loss=args.aux_loss,
            contrastive_align_loss=args.contrastive_align_loss,
            contrastive_hdim=args.contrastive_hdim,
            span_loss_type=args.span_loss_type,
            use_txt_pos=args.use_txt_pos,
            n_input_proj=args.n_input_proj,
            m_classes=args.m_classes,
            cls_both=args.cls_both, score_fg=args.score_fg,
            class_anchor=args.class_anchor,
            class_moe=args.class_moe, span_moe=args.span_moe,
            length_query=args.length_query
        )
    else:
        model = QDDETR(
            transformer,
            position_embedding,
            txt_position_embedding,
            txt_dim=args.t_feat_dim,
            vid_dim=args.v_feat_dim,
            aud_dim=args.a_feat_dim,
            num_queries=args.num_queries,
            input_dropout=args.input_dropout,
            aux_loss=args.aux_loss,
            contrastive_align_loss=args.contrastive_align_loss,
            contrastive_hdim=args.contrastive_hdim,
            span_loss_type=args.span_loss_type,
            use_txt_pos=args.use_txt_pos,
            n_input_proj=args.n_input_proj,
            m_classes=args.m_classes,
            cls_both=args.cls_both, score_fg=args.score_fg,
            class_anchor=args.class_anchor,
            class_moe=args.class_moe, span_moe=args.span_moe,
            length_query=args.length_query
        )

    matcher = build_matcher(args)
    weight_dict = {"loss_span": args.span_loss_coef,
                   "loss_giou": args.giou_loss_coef,
                   "loss_label": args.label_loss_coef,
                   "loss_saliency": args.lw_saliency}
    if args.contrastive_align_loss:
        weight_dict["loss_contrastive_align"] = args.contrastive_align_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items() if k != "loss_saliency"})
        weight_dict.update(aux_weight_dict)

    losses = ['spans', 'labels', 'saliency']
    if args.contrastive_align_loss:
        losses += ["contrastive_align"]
        
    # For tvsum dataset
    use_matcher = not (args.dset_name == 'tvsum')
        
    criterion = SetCriterion(
        matcher=matcher, weight_dict=weight_dict, losses=losses,
        eos_coef=args.eos_coef, temperature=args.temperature,
        span_loss_type=args.span_loss_type,  max_v_l=args.max_v_l,
        saliency_margin=args.saliency_margin, use_matcher=use_matcher,
        m_classes=args.m_classes,
        cls_both=args.cls_both, score_fg=args.score_fg, 
        label_loss_type=args.label_loss_type, focal_alpha=args.focal_alpha, focal_gamma=args.focal_gamma,
        aux_label_loss_type=args.aux_label_loss_type, aux_focal_alpha=args.aux_focal_alpha, aux_focal_gamma=args.aux_focal_gamma,
        length_span_weight=args.length_span_weight, length_giou_weight=args.length_giou_weight,
    )
    criterion.to(device)
    return model, criterion
