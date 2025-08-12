# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
import torch.nn.functional as F
from la_uvcom.span_utils import generalized_temporal_iou, span_cxw_to_xx

class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """
    def __init__(self,  cost_class: float = 1, cost_span: float = 1, cost_giou: float = 1,
                 span_loss_type: str = "l1", max_v_l: int = 75,
                 num_queries : int = 10, m_classes=None, cc_matcing=False,
                 class_anchor=False, tgt_embed=False, pos_query=1, length_query=None,):
        """Creates the matcher

        Params:
            cost_span: This is the relative weight of the L1 error of the span coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the spans in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_span = cost_span
        self.cost_giou = cost_giou
        self.span_loss_type = span_loss_type
        self.max_v_l = max_v_l
        self.foreground_label = 0
        assert cost_class != 0 or cost_span != 0 or cost_giou != 0, "all costs cant be 0"
        
        self.num_queries = num_queries

        self.m_classes = m_classes
        if m_classes is not None:
            self.num_classes = len(self.m_classes[1:-1].split(','))
        
        self.cc_matching = cc_matcing
        if self.cc_matching:
            assert m_classes is not None

            if not tgt_embed and not class_anchor:        
                self.num_queries = num_queries // self.num_classes

        self.pos_query = pos_query

        if length_query is not None:
            self.len_query_num = [int(lq) for lq in length_query[1:-1].split(',')]
        else:
            self.len_query_num = None

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_spans": Tensor of dim [batch_size, num_queries, 2] with the predicted span coordinates,
                    in normalized (cx, w) format
                 ""pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "spans": Tensor of dim [num_target_spans, 2] containing the target span coordinates. The spans are
                    in normalized (cx, w) format

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_spans)
        """

        if self.m_classes is not None and self.cc_matching:
            bs = outputs["pred_spans"].shape[0]

            batchwise_pred_indices = [ [] for i in range(bs)]
            batchwise_gt_indices = [ [] for i in range(bs)]

            cum_query_num = 0
            for cls_num in range(self.num_classes):

                gt_class_idx_align = []

                spans = []; target_sizes = []; org_target_sizes = []
                for i, v in enumerate(targets["moment_class"]): # batch : i
                    count = 0

                    sub_spans = []
                    sub_spans_idxs = []
                    for j, m in enumerate(v["m_cls"]):
                        if m == cls_num:
                            count += 1
                            sub_spans.append(targets["span_labels"][i]["spans"][j].unsqueeze(0))
                            sub_spans_idxs.append(j)
                    org_target_sizes.append(count)

                    idx_align = dict()

                    if count == 0: 
                        target_sizes.append(0)
                        idx_align = None
                        
                    elif count < self.pos_query:

                        mult = (math.ceil(self.pos_query / count))

                        for m in range(mult):
                            spans += sub_spans

                            for k in range(count):
                                idx_align[count * m + k] = sub_spans_idxs[k]

                        target_sizes.append(count * mult)

                    else:
                        for ii, sub_span_idx in enumerate(sub_spans_idxs):
                            idx_align[ii] = sub_span_idx
                        spans += sub_spans
                        target_sizes.append(count)

                    gt_class_idx_align.append(idx_align)

                if len(spans) == 0:
                    continue
                                    
                tgt_spans = torch.cat(spans)
                tgt_ids = torch.full([len(tgt_spans)], self.foreground_label)
                
                if self.len_query_num is None:
                    out_prob = outputs["pred_logits"][:, (self.num_queries*cls_num):self.num_queries*(cls_num+1), :].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]

                    tgt_ids = torch.full([len(tgt_spans)], self.foreground_label)   # [total #spans in the batch]
                    cost_class = -out_prob[:, tgt_ids] 
                    
                    if 'aux_pred_logits' in outputs.keys():
                        aux_out_prob = outputs["aux_pred_logits"][:, (self.num_queries*cls_num):self.num_queries*(cls_num+1), :].flatten(0, 1).softmax(-1)
                        aux_tgt_ids = torch.full([len(tgt_spans)], cls_num)
                        cost_class += (-aux_out_prob[:, aux_tgt_ids])
        
                    # We flatten to compute the cost matrices in a batch
                    out_spans = outputs["pred_spans"][:, (self.num_queries*cls_num):self.num_queries*(cls_num+1), :].flatten(0, 1)  # [batch_size * num_queries, 2]

                    # Compute the L1 cost between spans
                    cost_span = torch.cdist(out_spans, tgt_spans, p=1)  # [batch_size * num_queries, total #spans in the batch]

                    # Compute the giou cost between spans
                    # [batch_size * num_queries, total #spans in the batch]
                    cost_giou = - generalized_temporal_iou(span_cxw_to_xx(out_spans), span_cxw_to_xx(tgt_spans))

                    # Final cost matrix
                    # import ipdb; ipdb.set_trace()
                    C = self.cost_span * cost_span + self.cost_giou * cost_giou + self.cost_class * cost_class
                    C = C.view(bs, self.num_queries, -1).cpu()

                    # indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
                    # classwise_indices.append([(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices])
                    

                    for i, c in enumerate(C.split(target_sizes, -1)):
                        pred_temp_idxs, gt_temp_idxs = (linear_sum_assignment(c[i]))

                        for pred_temp_idx in pred_temp_idxs:
                            batchwise_pred_indices[i].append(pred_temp_idx + (self.num_queries * cls_num))

                        for gt_temp_idx in gt_temp_idxs:
                            batchwise_gt_indices[i].append(gt_class_idx_align[i][gt_temp_idx])
                else:
                    out_prob = outputs["pred_logits"][:, cum_query_num:cum_query_num+self.len_query_num[cls_num], :].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]

                    tgt_ids = torch.full([len(tgt_spans)], self.foreground_label)   # [total #spans in the batch]
                    cost_class = -out_prob[:, tgt_ids] 
                    
                    if 'aux_pred_logits' in outputs.keys():
                        aux_out_prob = outputs["aux_pred_logits"][:, cum_query_num:cum_query_num+self.len_query_num[cls_num], :].flatten(0, 1).softmax(-1)
                        aux_tgt_ids = torch.full([len(tgt_spans)], cls_num)
                        cost_class += (-aux_out_prob[:, aux_tgt_ids])
        
                    # We flatten to compute the cost matrices in a batch
                    out_spans = outputs["pred_spans"][:, cum_query_num:cum_query_num+self.len_query_num[cls_num], :].flatten(0, 1)  # [batch_size * num_queries, 2]

                    # Compute the L1 cost between spans
                    cost_span = torch.cdist(out_spans, tgt_spans, p=1)  # [batch_size * num_queries, total #spans in the batch]

                    # Compute the giou cost between spans
                    # [batch_size * num_queries, total #spans in the batch]
                    cost_giou = - generalized_temporal_iou(span_cxw_to_xx(out_spans), span_cxw_to_xx(tgt_spans))

                    # Final cost matrix
                    # import ipdb; ipdb.set_trace()
                    C = self.cost_span * cost_span + self.cost_giou * cost_giou + self.cost_class * cost_class
                    C = C.view(bs, self.len_query_num[cls_num], -1).cpu()

                    # indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
                    # classwise_indices.append([(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices])
                    

                    for i, c in enumerate(C.split(target_sizes, -1)):
                        pred_temp_idxs, gt_temp_idxs = (linear_sum_assignment(c[i]))

                        for pred_temp_idx in pred_temp_idxs:
                            batchwise_pred_indices[i].append(pred_temp_idx + cum_query_num)

                        for gt_temp_idx in gt_temp_idxs:
                            batchwise_gt_indices[i].append(gt_class_idx_align[i][gt_temp_idx])

                    cum_query_num += self.len_query_num[cls_num]

            final_indices = [(torch.as_tensor(p, dtype=torch.int64), torch.as_tensor(g, dtype=torch.int64)) for p, g in zip(batchwise_pred_indices, batchwise_gt_indices)]
        
            return final_indices
        
        bs, num_queries = outputs["pred_spans"].shape[:2]
        # targets = targets["span_labels"]

        # Also concat the target labels and spans
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        tgt_spans = torch.cat([v["spans"] for v in targets["span_labels"]])  # [num_target_spans in batch, 2]

        tgt_ids = torch.full([len(tgt_spans)], self.foreground_label)   # [total #spans in the batch]
        
        if 'aux_pred_logits' in outputs.keys():
            aux_out_prob = outputs["aux_pred_logits"].flatten(0, 1).softmax(-1)
            aux_tgt_ids = torch.cat([v["m_cls"] for v in targets['moment_class']])   # [total #spans in the batch]
                
        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - prob[target class].
        # The 1 is a constant that doesn't change the matching, it can be omitted.
        cost_class = -out_prob[:, tgt_ids]  # [batch_size * num_queries, total #spans in the batch]
        if 'aux_pred_logits' in outputs.keys():
            cost_class += (-aux_out_prob[:, aux_tgt_ids])

        if self.span_loss_type == "l1":
            # We flatten to compute the cost matrices in a batch
            out_spans = outputs["pred_spans"].flatten(0, 1)  # [batch_size * num_queries, 2]

            # Compute the L1 cost between spans
            cost_span = torch.cdist(out_spans, tgt_spans, p=1)  # [batch_size * num_queries, total #spans in the batch]

            # Compute the giou cost between spans
            # [batch_size * num_queries, total #spans in the batch]
            cost_giou = - generalized_temporal_iou(span_cxw_to_xx(out_spans), span_cxw_to_xx(tgt_spans))
        else:
            pred_spans = outputs["pred_spans"]  # (bsz, #queries, max_v_l * 2)
            pred_spans = pred_spans.view(bs * num_queries, 2, self.max_v_l).softmax(-1)  # (bsz * #queries, 2, max_v_l)
            cost_span = - pred_spans[:, 0][:, tgt_spans[:, 0]] - \
                pred_spans[:, 1][:, tgt_spans[:, 1]]  # (bsz * #queries, #spans)
            # pred_spans = pred_spans.repeat(1, n_spans, 1, 1).flatten(0, 1)  # (bsz * #queries * #spans, max_v_l, 2)
            # tgt_spans = tgt_spans.view(1, n_spans, 2).repeat(bs * num_queries, 1, 1).flatten(0, 1)  # (bsz * #queries * #spans, 2)
            # cost_span = pred_spans[tgt_spans]
            # cost_span = cost_span.view(bs * num_queries, n_spans)

            # giou
            cost_giou = 0

        # Final cost matrix
        # import ipdb; ipdb.set_trace()
        C = self.cost_span * cost_span + self.cost_giou * cost_giou + self.cost_class * cost_class
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["spans"]) for v in targets["span_labels"]]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(args):
    return HungarianMatcher(
        cost_span=args.set_cost_span, cost_giou=args.set_cost_giou,
        cost_class=args.set_cost_class, span_loss_type=args.span_loss_type, max_v_l=args.max_v_l,
        num_queries=args.num_queries, m_classes=args.m_classes, cc_matcing=args.cc_matching,
        tgt_embed=args.tgt_embed, class_anchor=args.class_anchor, pos_query=args.pos_query, length_query=args.length_query,
    )
