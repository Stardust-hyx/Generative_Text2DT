import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import BertConfig, BertPreTrainedModel
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_bert import BertEmbeddings, BertIntermediate, BertOutput, BertAttention
from models.modeling_cpt import CPTConfig, CPTPretrainedModel
from transformers.activations import ACT2FN
from copy import deepcopy


class EntityAwareBertConfig(CPTConfig):
    def __init__(self, num_generated_triples=None, entity_queries_num = None, mask_ent2tok = True, mask_tok2ent = False, mask_ent2ent = False, mask_entself = False,
    entity_aware_attention = False, entity_aware_intermediate = False, entity_aware_selfout = False, entity_aware_output = True, 
    use_entity_pos = True, use_entity_common_embedding = False,
    **kwargs):
        super(EntityAwareBertConfig, self).__init__( **kwargs)

        self.num_hidden_layers = self.encoder_layers
        self.intermediate_size = self.encoder_ffn_dim
        self.hidden_dropout_prob = self.dropout
        self.attention_probs_dropout_prob = self.attention_dropout
        self.layer_norm_eps = 1e-12
        self.hidden_act = "gelu"
        self.initializer_range = 0.02
        self.max_position_embeddings = 512

        self.num_generated_triples = num_generated_triples

        self.entity_queries_num = entity_queries_num
        self.mask_ent2tok = mask_ent2tok
        self.mask_tok2ent = mask_tok2ent
        self.mask_ent2ent = mask_ent2ent
        self.mask_entself = mask_entself
        self.entity_aware_attention = entity_aware_attention
        self.entity_aware_selfout = entity_aware_selfout
        self.entity_aware_intermediate = entity_aware_intermediate
        self.entity_aware_output = entity_aware_output
        self.type_vocab_size = 2

        self.use_entity_pos = use_entity_pos
        self.use_entity_common_embedding = use_entity_common_embedding


class EntityEmbeddings(nn.Module):
    def __init__(self, config, num_entry, is_pos_embedding=False):
        super().__init__()
        self.entity_embeddings = nn.Embedding(num_entry, config.hidden_size)
        self.use_common_embedding = config.use_entity_common_embedding and not is_pos_embedding
        if self.use_common_embedding:
            self.entity_common_embedding = nn.Embedding(1, config.hidden_size)
            self.register_buffer("common_index", torch.tensor(0))
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids):
        embeddings = self.entity_embeddings(input_ids)
        if self.use_common_embedding:
            embeddings = embeddings + self.entity_common_embedding(self.common_index)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class EntityAwareBertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.config = config
        
        
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        if config.entity_aware_attention:
            self.entity_e2w_query = nn.Linear(config.hidden_size, self.all_head_size)
            self.entity_e2w_key = nn.Linear(config.hidden_size, self.all_head_size)
            self.entity_e2w_value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, token_hidden_states, entity_hidden_states, attention_mask, query_pos = None):
        context_size = token_hidden_states.size(1)

        pos_aware_entity_hidden_states = entity_hidden_states
        if query_pos is not None:
            pos_aware_entity_hidden_states = (entity_hidden_states + query_pos)/2
            # pos_aware_entity_hidden_states = entity_hidden_states + query_pos
        # query specific
        w2w_query_layer = self.transpose_for_scores(self.query(token_hidden_states))

        if self.config.entity_aware_attention:
            e2w_query_layer = self.transpose_for_scores(self.entity_e2w_query(pos_aware_entity_hidden_states))
        else:
            e2w_query_layer = self.transpose_for_scores(self.query(pos_aware_entity_hidden_states))


        # key unified transformered
        w2w_key_layer = self.transpose_for_scores(self.key(token_hidden_states))

        if self.config.entity_aware_attention:
            e2w_key_layer = self.transpose_for_scores(self.entity_e2w_key(pos_aware_entity_hidden_states))
        else:
            e2w_key_layer = self.transpose_for_scores(self.key(pos_aware_entity_hidden_states))


        w2w_value_layer = self.transpose_for_scores(self.value(token_hidden_states))

        if self.config.entity_aware_attention:
            e2w_value_layer = self.transpose_for_scores(self.entity_e2w_value(entity_hidden_states))
        else:
            e2w_value_layer = self.transpose_for_scores(self.value(entity_hidden_states))

        w2w_attention_scores = torch.matmul(w2w_query_layer, w2w_key_layer.transpose(-1, -2))
        w2e_attention_scores = torch.matmul(w2w_query_layer, e2w_key_layer.transpose(-1, -2))
        e2w_attention_scores = torch.matmul(e2w_query_layer, w2w_key_layer.transpose(-1, -2))
        # w2e_attention_scores = torch.zeros(e2w_attention_scores.size()).transpose(-1, -2).to(e2w_attention_scores.device) - 1e30
        e2e_attention_scores = torch.matmul(e2w_query_layer, e2w_key_layer.transpose(-1, -2))

        word_attention_scores = torch.cat([w2w_attention_scores, w2e_attention_scores], dim=3)
        entity_attention_scores = torch.cat([e2w_attention_scores, e2e_attention_scores], dim=3)
        attention_scores = torch.cat([word_attention_scores, entity_attention_scores], dim=2)

        attention_scores = attention_scores / (self.attention_head_size**0.5)
        attention_scores = attention_scores + attention_mask

        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # value unified transformered
        value_layer = torch.cat([w2w_value_layer, e2w_value_layer], dim = -2)
        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer[:, :context_size, :], context_layer[:, context_size:, :]


class EntityAwareBertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        if config.entity_aware_selfout:
            self.entity_dense = nn.Linear(config.hidden_size, config.hidden_size)
            self.entity_LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, token_self_output, entity_self_output, token_hidden_states, entity_hidden_states):
        token_self_output = self.dense(token_self_output)
        token_self_output = self.dropout(token_self_output)
        token_self_output = self.LayerNorm(token_self_output + token_hidden_states)

        if self.config.entity_aware_selfout:
            entity_self_output = self.entity_dense(entity_self_output)
            entity_self_output = self.dropout(entity_self_output)
            entity_self_output = self.entity_LayerNorm(entity_self_output + entity_hidden_states)
        else:
            entity_self_output = self.dense(entity_self_output)
            entity_self_output = self.dropout(entity_self_output)
            entity_self_output = self.LayerNorm(entity_self_output + entity_hidden_states)
        hidden_states = torch.cat([token_self_output, entity_self_output], dim=1)
        return hidden_states


class EntityAwareBertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = EntityAwareBertSelfAttention(config)
        self.output = EntityAwareBertSelfOutput(config)

    def forward(self, word_hidden_states, entity_hidden_states, attention_mask, query_pos = None):
        word_self_output, entity_self_output = self.self(word_hidden_states, entity_hidden_states, attention_mask, query_pos = query_pos)
        output = self.output(word_self_output, entity_self_output, word_hidden_states, entity_hidden_states)
        return output[:, : word_hidden_states.size(1), :], output[:, word_hidden_states.size(1) :, :]

class EntityAwareBertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if config.entity_aware_intermediate:
            self.entity_dense = nn.Linear(config.hidden_size, config.intermediate_size)

        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, token_hidden_states, entity_hidden_states):
        token_hidden_states = self.dense(token_hidden_states)
        if self.config.entity_aware_intermediate:
            entity_hidden_states = self.entity_dense(entity_hidden_states)
        else:
            entity_hidden_states = self.dense(entity_hidden_states)

        token_hidden_states = self.intermediate_act_fn(token_hidden_states)
        entity_hidden_states = self.intermediate_act_fn(entity_hidden_states)

        return token_hidden_states, entity_hidden_states

class EntityAwareBertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        if self.config.entity_aware_output:
            self.entity_dense = nn.Linear(config.intermediate_size, config.hidden_size)
            self.entity_LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, token_intermediate_output, entity_intermediate_output, word_attention_output, entity_attention_output):
        token_intermediate_output = self.dense(token_intermediate_output)
        token_intermediate_output = self.dropout(token_intermediate_output)
        token_intermediate_output = self.LayerNorm(token_intermediate_output + word_attention_output)

        if self.config.entity_aware_output:
            entity_intermediate_output = self.entity_dense(entity_intermediate_output)
            entity_intermediate_output = self.dropout(entity_intermediate_output)
            entity_intermediate_output = self.entity_LayerNorm(entity_intermediate_output + entity_attention_output)
        else:
            entity_intermediate_output = self.dense(entity_intermediate_output)
            entity_intermediate_output = self.dropout(entity_intermediate_output)
            entity_intermediate_output = self.LayerNorm(entity_intermediate_output + entity_attention_output)

        return token_intermediate_output, entity_intermediate_output


class EntityAwareBertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.attention = EntityAwareBertAttention(config)
        self.intermediate = EntityAwareBertIntermediate(config)
        self.output = EntityAwareBertOutput(config)

    def forward(self, word_hidden_states, entity_hidden_states, attention_mask, query_pos = None):
        word_attention_output, entity_attention_output = self.attention(
            word_hidden_states, entity_hidden_states, attention_mask, query_pos = query_pos
        )

        token_intermediate_output, entity_intermediate_output = self.intermediate(word_attention_output, entity_attention_output)
        token_output, entity_output = self.output(token_intermediate_output, entity_intermediate_output, word_attention_output, entity_attention_output)

        return token_output, entity_output

class EntityAwareBertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList([EntityAwareBertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, word_hidden_states, entity_hidden_states, attention_mask, query_pos = None):
        intermediate = [{"h_token": word_hidden_states, "h_query": entity_hidden_states}]
        for layer_module in self.layer:
            word_hidden_states, entity_hidden_states = layer_module(
                word_hidden_states, entity_hidden_states, attention_mask, query_pos
            )
            intermediate.append({"h_token": word_hidden_states, "h_query": entity_hidden_states})
        return word_hidden_states, entity_hidden_states, intermediate


class EntitySelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.entity_attention = nn.MultiheadAttention(config.hidden_size, config.num_attention_heads, dropout=config.hidden_dropout_prob)
        self.entity_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.entity_norm = nn.LayerNorm(config.hidden_size)
    
    def forward(self, h_entity, query_pos):
        q = k = v = h_entity
        if query_pos is not None:
            q = k = h_entity + query_pos
        tgt = self.entity_attention(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1))[0].transpose(0, 1)
        tgt = h_entity + self.entity_dropout(tgt)
        h_entity = self.entity_norm(tgt)
        return h_entity

class SelfCrossAttention(nn.Module):
    def __init__(self, config, use_token_level_encoder, use_entity_attention, num_selfcrosslayer):
        super().__init__()
        self.use_token_level_encoder = use_token_level_encoder
        self.use_entity_attention = use_entity_attention
        self.num_selfcrosslayer = num_selfcrosslayer

        self.selflaters = None
        if self.use_entity_attention:
            self.selflaters = nn.ModuleList([EntitySelfAttention(config) for _ in range(num_selfcrosslayer)])

        self.crosslayers = None
        if self.use_token_level_encoder:
            self.crosslayers = nn.ModuleList([EntityAwareBertLayer(config) for _ in range(num_selfcrosslayer)])
    
    def forward(self, h_token, h_entity, token_entity_attention_mask, query_pos = None):
        intermediate = []
        for i in range(self.num_selfcrosslayer):

            if self.use_token_level_encoder:
                h_token, h_entity = self.crosslayers[i](h_token, h_entity, token_entity_attention_mask, query_pos = query_pos)

            if self.use_entity_attention:
                h_entity = self.selflaters[i](h_entity, query_pos)
            
            intermediate.append({"h_token":h_token, "h_query":h_entity})
                
        return h_token, h_entity, intermediate


class EntityBoundaryPredictor(nn.Module):
    def __init__(self, config, squeeze=True):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.token_embedding_linear = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size)
        ) 
        self.entity_embedding_linear = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size)
        ) 
        self.boundary_predictor = nn.Linear(self.hidden_size, 1)

        self.squeeze = squeeze
    
    def forward(self, token_embedding, entity_embedding, token_mask):
        # B x #ent x #token x hidden_size
        entity_token_matrix = self.token_embedding_linear(token_embedding).unsqueeze(1) + self.entity_embedding_linear(entity_embedding).unsqueeze(2)
        entity_token_cls = self.boundary_predictor(torch.relu(entity_token_matrix))
        token_mask = token_mask.unsqueeze(1).expand(-1, entity_token_cls.size(1), -1)
        if self.squeeze:
            entity_token_cls = entity_token_cls.squeeze(-1)
        else:
            token_mask = token_mask.unsqueeze(-1)
        entity_token_cls[~token_mask] = -10000
        entity_token_p = torch.sigmoid(entity_token_cls)

        return entity_token_cls, entity_token_p


class HeadTailBoundaryPredictor(nn.Module):
    def __init__(self, config, squeeze=True):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.token_embedding_linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.entity_embedding_linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.boundary_predictor = nn.Linear(self.hidden_size, 1, bias=False)
        self.squeeze = squeeze

        if squeeze:
            torch.nn.init.orthogonal_(self.token_embedding_linear.weight, gain=1)
            torch.nn.init.orthogonal_(self.entity_embedding_linear.weight, gain=1)
    
    def forward(self, token_embedding, entity_embedding, token_mask):
        # B x #ent x #token x hidden_size
        entity_token_matrix = self.token_embedding_linear(token_embedding).unsqueeze(1) + self.entity_embedding_linear(entity_embedding).unsqueeze(2)
        entity_token_cls = self.boundary_predictor(torch.relu(entity_token_matrix))
        token_mask = token_mask.unsqueeze(1).expand(-1, entity_token_cls.size(1), -1)
        if self.squeeze:
            entity_token_cls = entity_token_cls.squeeze(-1)
        else:
            token_mask = token_mask.unsqueeze(-1)
        entity_token_cls[~token_mask] = -10000
        entity_token_p = torch.sigmoid(entity_token_cls)

        return entity_token_cls, entity_token_p


class EntityTypePredictor(nn.Module):
    def __init__(self, config, cls_size, entity_type_count):
        super().__init__()

        
        self.linnear = nn.Linear(cls_size, config.hidden_size)

        self.multihead_attn = nn.MultiheadAttention(config.hidden_size, dropout=config.hidden_dropout_prob, num_heads=config.num_attention_heads)

        self.classifier = nn.Sequential(
            # nn.Linear(cls_size, config.hidden_size),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(config.hidden_size * 3, entity_type_count)
        )
    
    def forward(self, h_entity, h_token, p_left, p_right, token_mask):
        h_entity = self.linnear(torch.relu(h_entity))

        attn_output, _ = self.multihead_attn(h_entity.transpose(0, 1).clone(), h_token.transpose(0, 1), h_token.transpose(0, 1), key_padding_mask=~token_mask)
        attn_output = attn_output.transpose(0, 1)
        h_entity += attn_output
        
        left_token = torch.matmul(p_left, h_token)
        right_token = torch.matmul(p_right, h_token)

        h_entity = torch.cat([h_entity,left_token,right_token], dim =-1)
        
        entity_logits = self.classifier(h_entity)

        return entity_logits

class EntityAwareBertModel(BertPreTrainedModel):

    config_class = EntityAwareBertConfig
    def __init__(self, config):
        super().__init__(config)
        
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = EntityAwareBertEncoder(config)

        self.entity_embeddings = EntityEmbeddings(config, num_entry=config.entity_queries_num)
        self.triple_embeddings = EntityEmbeddings(config, num_entry=config.num_generated_triples)
        
        if config.use_entity_pos:
            self.pos_entity_embeddings = EntityEmbeddings(config, num_entry=config.entity_queries_num, is_pos_embedding=True)
            self.pos_triple_embeddings = EntityEmbeddings(config, num_entry=config.num_generated_triples, is_pos_embedding=True)

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _compute_extended_attention_mask(self, word_attention_mask: torch.LongTensor, entity_attention_mask: torch.LongTensor, mask_ent2tok = None, mask_tok2ent = None, mask_ent2ent = None, mask_entself = None, seg_mask = None):
        attention_mask = word_attention_mask
        if entity_attention_mask is not None:
            attention_mask = torch.cat([attention_mask, entity_attention_mask], dim=1)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        word_num = word_attention_mask.size(1)
        entity_num = entity_attention_mask.size(1)
        # #batch x #head x seq_len x seq_len
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2).expand(-1, -1, word_num+entity_num, -1).clone()

        # mask entity2token attention 
        if (mask_ent2tok == None and self.config.mask_ent2tok) or mask_ent2tok == True:
            extended_attention_mask[:, :, :word_num, word_num:] = 0
        
        if  (mask_tok2ent == None and self.config.mask_tok2ent) or mask_tok2ent == True:
            extended_attention_mask[:, :, word_num:, :word_num] = 0
        
        if seg_mask != None:
            tok2ent_mask = extended_attention_mask[:, :, word_num:, :word_num]
            seg_mask = seg_mask.bool().unsqueeze(1).unsqueeze(2).expand_as(tok2ent_mask)
            extended_attention_mask[:, :, word_num:, :word_num] = seg_mask

        if  (mask_ent2ent == None and self.config.mask_ent2ent) or mask_ent2ent == True:
            entity_attention = extended_attention_mask[:, :, word_num:, word_num:]
            mask = torch.eye(entity_num, entity_num, dtype = torch.bool).expand_as(entity_attention)
            entity_attention[~mask] = 0

        if (mask_entself == None and self.config.mask_entself) or mask_entself == True:
            entity_attention = extended_attention_mask[:, :, word_num:, word_num:]
            mask = torch.eye(entity_num, entity_num, dtype = torch.bool).expand_as(entity_attention)
            entity_attention[mask] = 0

        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        return extended_attention_mask

    def forward(self, token_input_ids, token_attention_mask, entity_ids, triple_ids, entity_triple_attention_mask = None, seg_encoding = None):
        word_embeddings = self.embeddings(token_input_ids, token_type_ids = seg_encoding)
            
        entity_embeddings = self.entity_embeddings(entity_ids)
        triple_embeddings = self.triple_embeddings(triple_ids)
        entity_triple_embeddings = torch.cat([entity_embeddings, triple_embeddings], dim=1)
        attention_mask = self._compute_extended_attention_mask(token_attention_mask, entity_triple_attention_mask, seg_mask=None)

        query_pos = None
        if self.config.use_entity_pos:
            ent_query_pos = self.pos_entity_embeddings(entity_ids)
            rel_query_pos = self.pos_triple_embeddings(triple_ids)
            query_pos = torch.cat([ent_query_pos, rel_query_pos], dim=1)

        return self.encoder(word_embeddings, entity_triple_embeddings, attention_mask, query_pos = query_pos)


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.crossattention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        encoder_attention_mask
    ):
        self_attention_outputs = self.attention(hidden_states)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
        encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        elif encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for encoder_hidden_shape (shape {}) or encoder_attention_mask (shape {})".format(
                    encoder_hidden_shape, encoder_attention_mask.shape
                )
            )
        encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -10000.0
        cross_attention_outputs = self.crossattention(
            hidden_states=attention_output, encoder_hidden_states=encoder_hidden_states,  encoder_attention_mask=encoder_extended_attention_mask
        )
        attention_output = cross_attention_outputs[0]
        outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs
        return outputs

class ProjectionHead(nn.Module):
    def __init__(self, dim_in, hidden_dim, feat_dim):
        super().__init__()
        self.net = nn.Sequential(
                nn.Linear(dim_in, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, feat_dim)
            )

    def forward(self, x):
        return self.net(x)


class PIQN(PreTrainedModel):

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def __init__(self, model_type, config: EntityAwareBertConfig, fix_bert_embeddings: bool, relation_type_count: int, entity_type_count: int, prop_drop: float, embed: torch.tensor = None, pos_size: int = 25, char_lstm_layers:int = 1, char_lstm_drop:int = 0.2, char_size:int = 25,  use_glove: bool = True, use_pos:bool = True, use_char_lstm:bool = True, lstm_layers = 3, pool_type:str = "max", word_mask_tok2ent = None, word_mask_ent2tok = None, word_mask_ent2ent = None, word_mask_entself = None, share_query_pos = False, use_token_level_encoder = True, num_token_ent_rel_layer = 2, num_token_ent_layer = 2, num_token_rel_layer = 1, num_token_head_tail_layer = 2, use_entity_attention = False, use_masked_lm = False, use_aux_loss = False, use_lstm = False, last_layer_for_loss = 3):
        super().__init__(config)

        self.model_type = model_type
        if model_type == "bert":
            self.bert = EntityAwareBertModel(config)
            self.model = self.bert
        else:
            self.encoder = EntityAwareBertModel(config)
            self.model = self.encoder

        self._keys_to_ignore_on_save = ["model." + k for k,v in self.model.named_parameters()]
        self._keys_to_ignore_on_load_unexpected = ["model." + k for k,v in self.model.named_parameters()]
        self._keys_to_ignore_on_load_missing = ["model." + k for k,v in self.model.named_parameters()]

        self.use_masked_lm = use_masked_lm
        self._relation_type_count = relation_type_count
        self._entity_type_count = entity_type_count
        self.prop_drop = prop_drop
        if embed is not None:
            self.wordvec_size = embed.size(-1)
        self.pos_size = pos_size
        self.use_glove = use_glove
        self.use_pos = use_pos
        self.char_lstm_layers = char_lstm_layers
        self.char_lstm_drop = char_lstm_drop
        self.char_size = char_size
        self.use_char_lstm = use_char_lstm
        self._share_query_pos = share_query_pos
        self.use_token_level_encoder = use_token_level_encoder
        self.num_token_ent_rel_layer = num_token_ent_rel_layer
        self.num_token_ent_layer = num_token_ent_layer
        self.num_token_rel_layer = num_token_rel_layer
        self.num_token_head_tail_layer = num_token_head_tail_layer
        self.use_entity_attention = use_entity_attention
        self.use_aux_loss = use_aux_loss
        self.use_lstm = use_lstm

        self.word_mask_tok2ent = word_mask_tok2ent
        self.word_mask_ent2tok = word_mask_ent2tok
        self.word_mask_ent2ent = word_mask_ent2ent
        self.word_mask_entself = word_mask_entself

        lstm_input_size = config.hidden_size

        if use_glove:
            lstm_input_size += self.wordvec_size
        if use_pos:
            lstm_input_size += self.pos_size
            self.pos_embedding = nn.Embedding(100, pos_size)
        if use_char_lstm:
            lstm_input_size += self.char_size * 2
            self.char_lstm = nn.LSTM(input_size = char_size, hidden_size = char_size, num_layers = char_lstm_layers,  bidirectional = True, dropout = char_lstm_drop, batch_first = True)
            self.char_embedding = nn.Embedding(103, char_size)

        if not self.use_lstm and (use_glove or use_pos or use_char_lstm):
            self.reduce_dimension = nn.Linear(lstm_input_size, config.hidden_size)

        if self.use_lstm:
            self.lstm = nn.LSTM(input_size = lstm_input_size, hidden_size = config.hidden_size//2, num_layers = lstm_layers,  bidirectional = True, dropout = 0.5, batch_first = True)

        self.pool_type = pool_type

        self.dropout = nn.Dropout(self.prop_drop)

        self.entity_classifier = EntityTypePredictor(config, config.hidden_size, entity_type_count + 1)
        # To try
        # self.entity_classifier = nn.Linear(config.hidden_size, entity_type_count + 1)

        self.left_boundary_classfier = EntityBoundaryPredictor(config)
        self.right_boundary_classfier = EntityBoundaryPredictor(config)

        if not self._share_query_pos and self.use_token_level_encoder:
            self.pos_entity_embeddings = EntityEmbeddings(config, num_entry=config.entity_queries_num, is_pos_embedding=True)
            self.pos_triple_embeddings = EntityEmbeddings(config, num_entry=config.num_generated_triples, is_pos_embedding=True)

        if self.use_token_level_encoder:
            self.selfcrossattention = SelfCrossAttention(config, use_token_level_encoder, use_entity_attention, num_token_ent_rel_layer)
            self.ent_layers = SelfCrossAttention(config, use_token_level_encoder, use_entity_attention, num_token_ent_layer)
            # self.rel_layers = SelfCrossAttention(config, use_token_level_encoder, use_entity_attention, num_token_rel_layer)
            # self.head_tail_layers = SelfCrossAttention(config, use_token_level_encoder, use_entity_attention, num_token_head_tail_layer)

        self.init_weights()

        if use_glove:
            self.wordvec_embedding = nn.Embedding.from_pretrained(embed, freeze=False)
        
        self.last_layer_for_loss = last_layer_for_loss


        self.register_buffer("entity_ids", torch.arange(self.config.entity_queries_num))
        self.register_buffer("triple_ids", torch.arange(self.config.num_generated_triples))
        self.register_buffer("entity_triple_attention_mask", torch.ones(self.config.entity_queries_num + self.config.num_generated_triples))
        self.register_buffer("entity_attention_mask", torch.ones(self.config.entity_queries_num))
        self.register_buffer("triple_attention_mask", torch.ones(self.config.num_generated_triples))
        self.register_buffer("head_tail_attention_mask", torch.ones(self.config.num_generated_triples * 2))

        if fix_bert_embeddings:
            self.model.embeddings.word_embeddings.weight.requires_grad = False
            self.model.embeddings.position_embeddings.weight.requires_grad = False
            self.model.embeddings.token_type_embeddings.weight.requires_grad = False

        self.ent_part_classifier = EntityBoundaryPredictor(config, squeeze=False)
        self.whether_ent_have_rel = nn.Linear(config.hidden_size, 2)

        self.Rel_layers = nn.ModuleList([DecoderLayer(config) for _ in range(num_token_rel_layer)])
        self.Head_Tail_layers = nn.ModuleList([DecoderLayer(config) for _ in range(num_token_head_tail_layer)])

        self.encode_trans_rel = nn.Linear(config.hidden_size, config.hidden_size)
        self.rel_classifier = nn.Linear(config.hidden_size, relation_type_count + 1)
        self.head_trans = nn.Linear(config.hidden_size, config.hidden_size)
        self.tail_trans = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.head_start_classifier = HeadTailBoundaryPredictor(config)
        self.head_end_classifier = HeadTailBoundaryPredictor(config)
        self.tail_start_classifier = HeadTailBoundaryPredictor(config)
        self.tail_end_classifier = HeadTailBoundaryPredictor(config)
        self.head_part_classifier = HeadTailBoundaryPredictor(config, squeeze=False)
        self.tail_part_classifier = HeadTailBoundaryPredictor(config, squeeze=False)

        self.head_type_classifier = nn.Linear(config.hidden_size, entity_type_count)
        self.tail_type_classifier = nn.Linear(config.hidden_size, entity_type_count)

        self.head_trans.weight.data.fill_(1)
        self.tail_trans.weight.data.fill_(1)

        # self.trans_triple  = nn.Linear(config.hidden_size, config.hidden_size)

        # For Contrastive Learning
        # self.ent_type_embeddings = nn.Embedding(entity_type_count, config.hidden_size)
        # self.rel_type_embeddings = nn.Embedding(relation_type_count, config.hidden_size)
        # self.project_entity = nn.Linear(config.hidden_size, 128)
        # self.project_rel = nn.Linear(config.hidden_size, 128)
        # self.project_head = nn.Linear(config.hidden_size, 128)
        # self.project_tail = nn.Linear(config.hidden_size, 128)

    def combine(self, sub, sup_mask, pool_type="max"):
        sup = None
        if len(sub.shape) == len(sup_mask.shape) :
            if pool_type == "mean":
                size = (sup_mask == 1).float().sum(-1).unsqueeze(-1) + 1e-30
                m = (sup_mask.unsqueeze(-1) == 1).float()
                sup = m * sub.unsqueeze(1).repeat(1, sup_mask.shape[1], 1, 1)
                sup = sup.sum(dim=2) / size
            if pool_type == "sum":
                m = (sup_mask.unsqueeze(-1) == 1).float()
                sup = m * sub.unsqueeze(1).repeat(1, sup_mask.shape[1], 1, 1)
                sup = sup.sum(dim=2)
            if pool_type == "max":
                m = (sup_mask.unsqueeze(-1) == 0).float() * (-1e30)
                sup = m + sub.unsqueeze(1).repeat(1, sup_mask.shape[1], 1, 1)
                sup = sup.max(dim=2)[0]
                sup[sup==-1e30]=0
        else:
            if pool_type == "mean":
                size = (sup_mask == 1).float().sum(-1).unsqueeze(-1) + 1e-30
                m = (sup_mask.unsqueeze(-1) == 1).float()
                sup = m * sub
                sup = sup.sum(dim=2) / size
            if pool_type == "sum":
                m = (sup_mask.unsqueeze(-1) == 1).float()
                sup = m * sub
                sup = sup.sum(dim=2)
            if pool_type == "max":
                m = (sup_mask.unsqueeze(-1) == 0).float() * (-1e30)
                sup = m + sub
                sup = sup.max(dim=2)[0]
                sup[sup==-1e30]=0
        return sup

    def forward(self, encodings:torch.tensor, context_masks:torch.tensor, seg_encoding:torch.tensor, context2token_masks:torch.tensor, token_masks:torch.tensor, sent_regions=None,
        pos_encoding: torch.tensor = None, wordvec_encoding:torch.tensor = None, char_encoding:torch.tensor = None, token_masks_char = None, char_count:torch.tensor = None):
        
        batch_size = encodings.shape[0]
        token_count = token_masks.long().sum(-1,keepdim=True)

        context_masks = context_masks.float()
        token_masks_float = token_masks.float()

        entity_ids = self.entity_ids.expand(batch_size, -1)
        triple_ids = self.triple_ids.expand(batch_size, -1)
        entity_triple_attention_mask = self.entity_triple_attention_mask.expand(batch_size, -1)

        h, h_entity_triple, _ = self.model(token_input_ids=encodings, token_attention_mask=context_masks,
                                    entity_ids=entity_ids, triple_ids=triple_ids, entity_triple_attention_mask=entity_triple_attention_mask, seg_encoding=seg_encoding)
        
        if context2token_masks:
            h_token = self.combine(h, context2token_masks, self.pool_type)
        else:
            h_token = h

        if self.use_lstm:
            h_token = nn.utils.rnn.pack_padded_sequence(input = h_token, lengths = token_count.squeeze(-1).cpu().tolist(), enforce_sorted = False, batch_first = True)
            h_token, (_, _) = self.lstm(h_token)
            h_token, _ = nn.utils.rnn.pad_packed_sequence(h_token, batch_first=True)


        token_entity_triple_attention_mask = self.model._compute_extended_attention_mask(token_masks, entity_triple_attention_mask, mask_ent2tok =  self.word_mask_ent2tok, mask_tok2ent =  self.word_mask_tok2ent, mask_ent2ent =  self.word_mask_ent2ent, mask_entself = self.word_mask_entself)

        entity_attention_mask = self.entity_attention_mask.expand(batch_size, -1)
        triple_attention_mask = self.triple_attention_mask.expand(batch_size, -1)
        head_tail_attention_mask = self.head_tail_attention_mask.expand(batch_size, -1)
        token_entity_attention_mask = self.model._compute_extended_attention_mask(token_masks, entity_attention_mask, mask_ent2tok =  self.word_mask_ent2tok, mask_tok2ent =  self.word_mask_tok2ent, mask_ent2ent =  self.word_mask_ent2ent, mask_entself = self.word_mask_entself)
        token_triple_attention_mask = self.model._compute_extended_attention_mask(token_masks, triple_attention_mask, mask_ent2tok =  self.word_mask_ent2tok, mask_tok2ent =  self.word_mask_tok2ent, mask_ent2ent =  self.word_mask_ent2ent, mask_entself = self.word_mask_entself)
        token_head_tail_attention_mask = self.model._compute_extended_attention_mask(token_masks, head_tail_attention_mask, mask_ent2tok =  self.word_mask_ent2tok, mask_tok2ent =  self.word_mask_tok2ent, mask_ent2ent =  self.word_mask_ent2ent, mask_entself = self.word_mask_entself)

        query_pos = None
        if self.config.use_entity_pos and self._share_query_pos and self.use_token_level_encoder:
            ent_query_pos = self.model.pos_entity_embeddings(entity_ids)
            rel_query_pos = self.model.pos_triple_embeddings(triple_ids)
        
        if not self._share_query_pos and self.use_token_level_encoder:
            ent_query_pos = self.pos_entity_embeddings(entity_ids)
            rel_query_pos = self.pos_triple_embeddings(triple_ids)

        query_pos = torch.cat([ent_query_pos, rel_query_pos], dim=1)
 
        h_token, h_entity_triple, _ = self.selfcrossattention(h_token, h_entity_triple, token_entity_triple_attention_mask, query_pos=query_pos)
        h_entity, h_triple = torch.split(h_entity_triple, [self.config.entity_queries_num, self.config.num_generated_triples], dim=1)

        h_token_ent_view, h_entity_, ent_intermediate = self.ent_layers(h_token, h_entity, token_entity_attention_mask, query_pos=ent_query_pos)

        # h_token_rel_view, h_triple, rel_intermediate = self.rel_layers(h_token, h_triple, token_triple_attention_mask, query_pos=rel_query_pos)
        for i, layer_module in enumerate(self.Rel_layers):
            layer_outputs = layer_module(
                h_triple, h_token, token_masks_float
            )
            h_triple = layer_outputs[0]

        h_head_tail = torch.repeat_interleave(h_triple, 2, dim=1)
        h_head_tail[:, ::2, :] = self.head_trans(h_triple)
        h_head_tail[:, 1::2, :] = self.tail_trans(h_triple)
        h_head_tail = self.LayerNorm(h_head_tail)

        # h_token_head_tail_view, h_head_tail, head_tail_intermediate = self.head_tail_layers(h_token_rel_view, h_head_tail, token_head_tail_attention_mask)
        h_token_rel_view = self.encode_trans_rel(h_token)
        for i, layer_module in enumerate(self.Head_Tail_layers):
            layer_outputs = layer_module(
                h_head_tail, h_token_rel_view, token_masks_float
            )
            h_head_tail = layer_outputs[0]

        ent_outputs = []
        rel_outputs = []
        for h_dict in ent_intermediate:
            _h_token, _h_entity = h_dict["h_token"], h_dict["h_query"]

            logit_left, p_left = self.left_boundary_classfier(_h_token, _h_entity, token_masks)
            logit_right, p_right = self.right_boundary_classfier(_h_token, _h_entity, token_masks)
            entity_logits = self.entity_classifier(_h_entity, _h_token, p_left, p_right, token_masks)
            logit_part, p_part = self.ent_part_classifier(_h_token, _h_entity, token_masks) if self.training else (None, None)
            ent_have_rel_logits = self.whether_ent_have_rel(_h_entity) if self.training else None            
            ent_outputs.append({"ent_type_logits": entity_logits, "ent_start_logits": logit_left, "ent_end_logits": logit_right, "ent_part_logits": logit_part,
                                "p_left": p_left, "p_right": p_right, "p_part": p_part, "ent_have_rel_logits": ent_have_rel_logits})

        class_logits = self.rel_classifier(h_triple)
        h_head = h_head_tail[:, ::2, :]
        h_tail = h_head_tail[:, 1::2, :]
        head_start_logits, p_head_start = self.head_start_classifier(h_token_rel_view, h_head, token_masks)
        head_end_logits, p_head_end = self.head_end_classifier(h_token_rel_view, h_head, token_masks)
        tail_start_logits, p_tail_start = self.tail_start_classifier(h_token_rel_view, h_tail, token_masks)
        tail_end_logits, p_tail_end = self.tail_end_classifier(h_token_rel_view, h_tail, token_masks)
        head_part_logits, p_head_part = self.head_part_classifier(h_token_rel_view, h_head, token_masks)
        tail_part_logits, p_tail_part = self.tail_part_classifier(h_token_rel_view, h_tail, token_masks)
        head_type_logits = self.head_type_classifier(h_head)
        tail_type_logits = self.tail_type_classifier(h_tail)
        rel_outputs.append({'pred_rel_logits': class_logits, 'head_start_logits': head_start_logits, 'head_end_logits': head_end_logits,
                            'tail_start_logits': tail_start_logits, 'tail_end_logits': tail_end_logits,
                            'head_part_logits': head_part_logits, 'tail_part_logits': tail_part_logits,
                            'head_type_logits': head_type_logits, 'tail_type_logits': tail_type_logits,
                            'p_head_start': p_head_start, 'p_head_end': p_head_end, 'p_tail_start': p_tail_start, 'p_tail_end': p_tail_end,
                            'p_head_part': p_head_part, 'p_tail_part': p_tail_part})

        project_entity = None
        # project_entity = self.project_entity(h_entity_)
        # project_rel = self.project_rel(h_triple)
        # project_head = self.project_head(h_head)
        # project_tail = self.project_tail(h_tail)
        # project_triple = project_rel + project_head + project_tail
        project_triple = h_tail
        # project_triple = self.trans_triple (project_triple)

        context = h_token_rel_view
        # context = torch.where(h_token > h_token_rel_view, h_token, h_token_rel_view)
        # context = torch.where(h > context, h, context)

        return context, ent_outputs, rel_outputs, project_entity, project_triple


class SeqEncoder(PIQN):
    
    config_class = CPTConfig
    base_model_prefix = "model.encoder"
    authorized_missing_keys = [r"position_ids"]

    def __init__(self, *args, **kwagrs):
        super().__init__("CPT", *args, **kwagrs)

