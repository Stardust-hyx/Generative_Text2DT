"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn

class TripleMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, loss_weight, matcher, boundary_softmax=False):
        super().__init__()
        # self.cost_relation = 1.0
        # self.cost_head = 2.0
        # self.cost_tail = 2.0

        self.cost_relation = loss_weight["relation"]
        self.cost_head = loss_weight["head_entity"]
        self.cost_tail = loss_weight["tail_entity"]

        self.matcher = matcher
        self.boundary_softmax = boundary_softmax

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_rel_logits": Tensor of dim [batch_size, num_generated_triples, num_classes] with the classification logits
                 "{head, tail}_{start, end}_logits": Tensor of dim [batch_size, num_generated_triples, seq_len] with the predicted index logits
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_generated_triples, num_gold_triples)
        """
        bsz, num_generated_triples = outputs["pred_rel_logits"].shape[:2]
        # We flatten to compute the cost matrices in a batch
        pred_rel = outputs["pred_rel_logits"].flatten(0, 1).softmax(-1)  # [bsz * num_generated_triples, num_classes]
        gold_rel = torch.cat([v["relation"] for v in targets])
        # after masking the pad token
        if self.boundary_softmax:
            pred_head_start = outputs["head_start_logits"].flatten(0, 1).softmax(-1)  # [bsz * num_generated_triples, seq_len]
            pred_head_end = outputs["head_end_logits"].flatten(0, 1).softmax(-1)
            pred_tail_start = outputs["tail_start_logits"].flatten(0, 1).softmax(-1)
            pred_tail_end = outputs["tail_end_logits"].flatten(0, 1).softmax(-1)
        else:
            pred_head_start = outputs["p_head_start"].flatten(0, 1)  # [bsz * num_generated_triples, seq_len]
            pred_head_end = outputs["p_head_end"].flatten(0, 1)
            pred_tail_start = outputs["p_tail_start"].flatten(0, 1)
            pred_tail_end = outputs["p_tail_end"].flatten(0, 1)

        gold_head_start = torch.cat([v["head_start_index"] for v in targets])
        gold_head_end = torch.cat([v["head_end_index"] for v in targets])
        gold_tail_start = torch.cat([v["tail_start_index"] for v in targets])
        gold_tail_end = torch.cat([v["tail_end_index"] for v in targets])
        if self.matcher == "avg":
            cost = - self.cost_relation * pred_rel[:, gold_rel] - self.cost_head * 1/2 * (pred_head_start[:, gold_head_start] + pred_head_end[:, gold_head_end]) - self.cost_tail * 1/2 * (pred_tail_start[:, gold_tail_start] + pred_tail_end[:, gold_tail_end])
        elif self.matcher == "min":
            cost = torch.cat([pred_head_start[:, gold_head_start].unsqueeze(1), pred_rel[:, gold_rel].unsqueeze(1), pred_head_end[:, gold_head_end].unsqueeze(1), pred_tail_start[:, gold_tail_start].unsqueeze(1), pred_tail_end[:, gold_tail_end].unsqueeze(1)], dim=1)
            cost = - torch.min(cost, dim=1)[0]
        else:
            raise ValueError("Wrong matcher")
        cost = cost.view(bsz, num_generated_triples, -1).cpu()
        num_gold_triples = [len(v["relation"]) for v in targets]
        # print('num_gold_triples', num_gold_triples)
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost.split(num_gold_triples, -1))]
        rel_indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

        # for indice in rel_indices:
        #     print(indice)

        return rel_indices


class EntityMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, loss_weight, matcher, boundary_softmax=False):
        super().__init__()
        self.cost_ent_type = loss_weight["ent_type"]
        self.cost_ent_span = loss_weight["ent_span"]

        self.matcher = matcher
        self.boundary_softmax = boundary_softmax

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_rel_logits": Tensor of dim [batch_size, num_generated_triples, num_classes] with the classification logits
                 "{head, tail}_{start, end}_logits": Tensor of dim [batch_size, num_generated_triples, seq_len] with the predicted index logits
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_generated_triples, num_gold_triples)
        """

        bsz, num_generated_entities = outputs["ent_type_logits"].shape[:2]

        pred_ent_type = outputs["ent_type_logits"].flatten(0, 1).softmax(-1)  # [bsz * num_generated_entities, num_ent_types]
        gold_ent_type = torch.cat([v["ent_type"] for v in targets])

        if self.boundary_softmax:
            pred_ent_start = outputs["ent_start_logits"].flatten(0, 1).softmax(-1)  # [bsz * num_generated_entities, seq_len]
            pred_ent_end = outputs["ent_end_logits"].flatten(0, 1).softmax(-1)
        else:
            pred_ent_start = outputs["p_left"].flatten(0, 1)
            pred_ent_end = outputs["p_right"].flatten(0, 1)

        gold_ent_start = torch.cat([v["ent_start_index"] for v in targets])
        gold_ent_end = torch.cat([v["ent_end_index"] for v in targets])

        cost = - self.cost_ent_type * pred_ent_type[:, gold_ent_type] - self.cost_ent_span * (pred_ent_start[:, gold_ent_start] + pred_ent_end[:, gold_ent_end])
        cost = cost.view(bsz, num_generated_entities, -1).cpu()
        num_gold_entities = [len(v["ent_type"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost.split(num_gold_entities, -1))]
        ent_indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

        return ent_indices

class Ent_HeadTail_Matcher(nn.Module):
    """ 
    """

    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, ent_start_probs, ent_end_probs, ent_part_probs, target_start_probs, target_end_probs, target_part_probs,
                ent_type_probs=None, target_type_probs=None):
        """ Performs the matching
        """
        list_indices = []
        for i, (target_start_prob, target_end_prob, target_part_prob) in enumerate(zip(target_start_probs, target_end_probs, target_part_probs)):
            ent_start_prob = ent_start_probs[i]
            ent_end_prob = ent_end_probs[i]
            ent_part_prob = ent_part_probs[i]

            if target_type_probs is not None:
                target_type_prob = target_type_probs[i]
                ent_type_prob = ent_type_probs[i]

            num_pred_ent = ent_start_prob.size(0)
            num_target_ent = target_start_prob.size(0)

            # print(ent_start_prob.shape, flush=True)
            # print(target_start_prob.shape, flush=True)
            # print()

            if num_pred_ent == 0 or num_target_ent == 0:
                indices = None

            else:
                ent_start_prob = ent_start_prob.unsqueeze(0).expand(num_target_ent, -1, -1)
                ent_end_prob = ent_end_prob.unsqueeze(0).expand(num_target_ent, -1, -1)
                ent_part_prob = ent_part_prob.unsqueeze(0).expand(num_target_ent, -1, -1)

                target_start_prob = target_start_prob.unsqueeze(1).expand(-1, num_pred_ent, -1)
                target_end_prob = target_end_prob.unsqueeze(1).expand(-1, num_pred_ent, -1)
                target_part_prob = target_part_prob.unsqueeze(1).expand(-1, num_pred_ent, -1)

                # print(ent_start_prob.shape, flush=True)
                # print(target_start_prob.shape, flush=True)

                start_cost = F.kl_div(ent_start_prob, target_start_prob, reduction='none', log_target=True)
                end_cost = F.kl_div(ent_end_prob, target_end_prob, reduction='none', log_target=True)
                
                cost = start_cost.sum(-1) + end_cost.sum(-1)

                if target_type_probs is not None:
                    ent_type_prob = ent_type_prob.unsqueeze(0).expand(num_target_ent, -1, -1)
                    target_type_prob = target_type_prob.unsqueeze(1).expand(-1, num_pred_ent, -1)
                    type_cost = F.kl_div(ent_type_prob, target_type_prob, reduction='none', log_target=True)
                    cost += type_cost.sum(-1)

                # print(cost[:3, :4], flush=True)
                indices = cost.min(-1).indices
            
            list_indices.append(indices)
            # print('indices.shape', indices.shape, flush=True)

        return list_indices
