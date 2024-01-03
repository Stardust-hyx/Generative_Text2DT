import torch, random, gc, os, json
from torch import nn, optim
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from transformers import AdamW
from datetime import timedelta, datetime
from collections import defaultdict
from utils.average_meter import AverageMeter
from utils.functions import formulate_gold_, formulate_gold_ent
from utils.metric import metric_, ent_metric
from text2dt_eval.text2dt_metric import text2dt_metric

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def get_linear_schedule_with_warmup_two_stage(optimizer, num_warmup_steps_stage_one, num_training_steps_stage_one, num_warmup_steps_stage_two, num_training_steps_stage_two,  last_epoch=-1):
    def lr_lambda(current_step: int):
        if current_step < num_training_steps_stage_one:
            if current_step < num_warmup_steps_stage_one:
                return float(current_step) / float(max(1, num_warmup_steps_stage_one))
            return max(
                0.0, float(num_training_steps_stage_one - current_step) / float(max(1, num_training_steps_stage_one - num_warmup_steps_stage_one))
            )
        else:
            current_step = current_step - num_training_steps_stage_one
            if current_step < num_warmup_steps_stage_two:
                return float(current_step) / float(max(1, num_warmup_steps_stage_two))
            return max(
                0.0, float(num_training_steps_stage_two - current_step) / float(max(1, num_training_steps_stage_two - num_warmup_steps_stage_two))
            )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


class Trainer(nn.Module):
    def __init__(self, model, data, args, max_epoch, start_eval_epoch, gen):
        super().__init__()
        self.args = args
        self.model = model
        self.data = data

        self.max_epoch = max_epoch
        self.start_eval_epoch = start_eval_epoch
        self.gen = gen

        self.save_model = args.save_model
        os.makedirs(args.checkpoint_directory, exist_ok=True)

        # pre = ''
        # for n, p in self.model.named_parameters():
        #     cur = '.'.join(n.split('.')[:2])
        #     if pre != cur:
        #         pre = cur
        #         print()
        #     print(n)

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

        grouped_params = [
            {
                'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)
                            and 'PIQN.encoder' in n ],
                'weight_decay': args.weight_decay,
                'lr': args.encoder_lr,
            },
            {
                'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)
                            and 'PIQN.encoder' in n],
                'weight_decay': 0.0,
                'lr': args.encoder_lr,
            },
            {
                'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay) 
                            and 'PIQN.encoder' not in n],
                'weight_decay': args.weight_decay,
                'lr': args.decoder_lr,
            },
            {
                'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay) 
                            and 'PIQN.encoder' not in n],
                'weight_decay': 0.0,
                'lr': args.decoder_lr,
            }
        ]

        # print([n for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)
        #                     and 'PIQN.encoder' in n])
        # print()
        # print([n for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay) 
        #                     and 'PIQN.encoder' not in n])
        
        if args.optimizer == 'Adam':
            self.optimizer = optim.Adam(grouped_params)
        elif args.optimizer == 'AdamW':
            self.optimizer = AdamW(grouped_params)
        else:
            raise Exception("Invalid optimizer.")
        if args.use_gpu:
            self.cuda()

    def train_model(self):
        best_f1 = 0
        train_loader = self.data.train_loader
        train_num = len(train_loader)
        batch_size = self.args.batch_size
        total_batch = train_num // batch_size + 1
        updates_total = total_batch * self.max_epoch
        updates_total_stage_one = total_batch * self.args.split_epoch
        updates_total_stage_two = total_batch * (self.max_epoch - self.args.split_epoch)
        scheduler = get_linear_schedule_with_warmup_two_stage(self.optimizer,
                                                    num_warmup_steps_stage_one = self.args.lr_warmup * updates_total_stage_one,
                                                    num_training_steps_stage_one = updates_total_stage_one,
                                                    num_warmup_steps_stage_two = self.args.lr_warmup * updates_total_stage_two,
                                                    num_training_steps_stage_two = updates_total_stage_two)


        valid_gold = json.load(open(self.args.valid_json, "r"))
        test_gold = json.load(open(self.args.test_json, "r"))

        start_datetime_str = datetime.now().strftime('%m-%d-%H-%M-%S')
        if self.gen:
            print('\n----------- Start Gen Training -----------', start_datetime_str)
        else:
            print('\n----------- Start RE Training -----------', start_datetime_str)
        for epoch in range(self.max_epoch):
            # Train
            self.model.train()
            self.model.zero_grad()

            if not self.gen:
                if epoch == 0:
                    print('Freeze Decoder.')
                    for name, param in self.model.decoder.named_parameters():
                        param.requires_grad = False

                    print("Freeze bert weights")
                    for name, param in self.model.PIQN.model.named_parameters():
                        if "entity" not in name and "triple" not in name:
                            param.requires_grad = False

                if epoch == self.args.split_epoch:
                    print("Now, update bert weights.")
                    for name, param in self.model.PIQN.model.named_parameters():
                        param.requires_grad = True
                    if self.args.fix_bert_embeddings:
                        self.model.PIQN.model.embeddings.word_embeddings.weight.requires_grad = False
                        self.model.PIQN.model.embeddings.position_embeddings.weight.requires_grad = False
                        self.model.PIQN.model.embeddings.token_type_embeddings.weight.requires_grad = False

                    self.optimizer.__setstate__({'state': defaultdict(dict)})

            else:
                if epoch == 0:
                    print('Unfreeze Decoder.')
                    for name, param in self.model.decoder.named_parameters():
                        param.requires_grad = True

                    print('Unfreeze Encoder.')
                    for name, param in self.model.PIQN.named_parameters():
                        param.requires_grad = True
                    if self.args.fix_bert_embeddings:
                        self.model.PIQN.model.embeddings.word_embeddings.weight.requires_grad = False
                        self.model.PIQN.model.embeddings.position_embeddings.weight.requires_grad = False
                        self.model.PIQN.model.embeddings.token_type_embeddings.weight.requires_grad = False

            print("=== Epoch %d train ===" % epoch, flush=True)
            avg_loss = AverageMeter()
            random.shuffle(train_loader)
            for batch_id in range(total_batch):
                start = batch_id * batch_size
                end = (batch_id + 1) * batch_size
                if end > train_num:
                    end = train_num
                train_instance = train_loader[start:end]
                # print([ele[0] for ele in train_instance])
                if not train_instance:
                    continue
                input_ids, attention_mask, seg_encoding, context2token_masks, token_masks, targets, tgt_seq_ids, tgt_seq_len, info = \
                        self.model.batchify(train_instance)
                loss = self.model.forward(input_ids, attention_mask, seg_encoding, context2token_masks, token_masks, targets, tgt_seq_ids, tgt_seq_len, info['relID2triples'], 
                                          epoch=epoch, gen=self.gen)[0]
                avg_loss.update(loss.item(), 1)
                # Optimize
                loss.backward()
                if self.args.max_grad_norm != 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                if (batch_id + 1) % self.args.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    scheduler.step()
                    self.model.zero_grad()
            # for param_group in self.optimizer.param_groups:
            #     print(param_group['lr'])

            print("     Instance: %d; loss: %.4f" % (end, avg_loss.avg), flush=True)
            if epoch >= self.start_eval_epoch:
                # Validation
                print("=== Epoch %d Validation ===" % epoch)
                result = self.eval_model(self.data.valid_loader, self.data.relational_alphabet, valid_gold)
                f1 = (result['path_f1'] + result['tree_acc']) if 'path_f1' in result else result['f1']
                if f1 >= best_f1:
                    print("Achieving Best Result on Validation Set.", flush=True)
                    if self.save_model:
                        torch.save(self.model.state_dict(), self.args.checkpoint_directory + "/%s.model" % start_datetime_str)
                    best_f1 = f1
                    best_result_epoch = epoch
                    # # Test
                    # print("=== Epoch %d Test ===" % epoch, flush=True)
                    # result = self.eval_model(self.data.test_loader, self.data.relational_alphabet, test_gold)


        end_datetime_str = datetime.now().strftime('%m-%d-%H-%M-%S')
        if self.gen:
            print('\n----------- Finish Gen Training -----------', end_datetime_str)
        else:
            print('\n----------- Finish RE Training -----------', end_datetime_str)
        # if self.save_model:
        #     torch.save(self.model.state_dict(), self.args.checkpoint_directory + "/%s.model" % end_datetime_str)
        print("Best result on validation set achieved at epoch %d." % best_result_epoch, flush=True)
        print("=== Final Test === ", flush=True)
        self.load_state_dict(self.args.checkpoint_directory + "/%s.model" % start_datetime_str)
        result = self.eval_model(self.data.test_loader, self.data.relational_alphabet, test_gold,
                                log_fn=os.path.join(self.args.checkpoint_directory, start_datetime_str), print_pred=self.args.print_pred)
        # self.load_state_dict(self.args.checkpoint_directory + "/%s.model" % end_datetime_str)
        # result = self.eval_model(self.data.test_loader, self.data.relational_alphabet, test_gold)


    def eval_model(self, eval_loader, relational_alphabet, gold_data, log_fn=None, print_pred=False):
        self.model.eval()
        # print(self.model.decoder.query_embed.weight)
        prediction, gold_ = {}, {}
        prediction_ent, gold_ent = {}, {}
        list_text = []
        list_res = []
        with torch.no_grad():
            batch_size = self.args.batch_size
            eval_num = len(eval_loader)
            total_batch = eval_num // batch_size + 1
            for batch_id in range(total_batch):
                start = batch_id * batch_size
                end = (batch_id + 1) * batch_size
                if end > eval_num:
                    end = eval_num
                eval_instance = eval_loader[start:end]
                if not eval_instance:
                    continue
                input_ids, attention_mask, seg_encoding, context2token_masks, token_masks, target, _, _, info = self.model.batchify(eval_instance, is_test=True)
                # print(target)
                gold_.update(formulate_gold_(target, info, relational_alphabet))
                # gold_ent.update(formulate_gold_ent(target, info))
                gen_triples, gen_entities, res = self.model.predict(input_ids, attention_mask, seg_encoding, context2token_masks, token_masks, info, gen=self.gen)
                prediction.update(gen_triples)
                # prediction_ent.update(gen_entities)
                list_text += info['text']
                list_res += res

        triple_score = metric_(prediction, gold_, list_text, relational_alphabet, log_fn, print_pred)

        if not self.gen:
            return triple_score
        
        if print_pred:
            f = open(log_fn, 'w')
            json.dump(list_res, f, ensure_ascii=False, indent=2)

        return text2dt_metric(gold_data, list_res)


    def load_state_dict(self, path):
        self.model.load_state_dict(torch.load(path))

    @staticmethod
    def lr_decay(optimizer, epoch, decay_rate):
        # lr = init_lr * ((1 - decay_rate) ** epoch)
        if epoch != 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * (1 - decay_rate)
                # print(param_group['lr'])
        return optimizer
