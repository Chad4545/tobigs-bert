from bert import *
from bert.utils import *
from bert.classify import *
from bert.pretrain import *

def dataset_class():
    # TODO: 함수명 변경 및 main 함수에서도 변경
    # label, sentence
    pass

# TODO: embedding layer 수정한 pipeline 만들기


class Binary(CsvDataset):
    """ Dataset class for MNLI """
    labels = ('0', '1') # label names
    def __init__(self, file, pipeline=[]):
        super().__init__(file, pipeline)

    def get_instances(self, lines):
        for line in itertools.islice(lines, 1, None): # skip header
            yield line[0], line[1], [] # label, text


def pretrain(train_cfg='config/pretrain.json',
         model_cfg='config/bert_base.json',
         data_file='data/bert_pretrain.txt',
         model_file=None,
         data_parallel=True,
         vocab='data/vocab.txt',
         save_dir='pretrain',
         log_dir='pretrain/runs',
         max_len=64,
         max_pred=20,
         mask_prob=0.15):

    cfg = train.Config.from_json(train_cfg)
    model_cfg = models.Config.from_json(model_cfg)

    set_seeds(cfg.seed)

    tokenizer = tokenization.FullTokenizer(vocab_file=vocab, do_lower_case=True)
    tokenize = lambda x: tokenizer.tokenize(tokenizer.convert_to_unicode(x))

    pipeline = [Preprocess4Pretrain(max_pred,
                                    mask_prob,
                                    list(tokenizer.vocab.keys()),
                                    tokenizer.convert_tokens_to_ids,
                                    max_len)]
    data_iter = SentPairDataLoader(data_file,
                                   cfg.batch_size,
                                   tokenize,
                                   max_len,
                                   pipeline=pipeline)

    model = BertModel4Pretrain(model_cfg)
    criterion1 = nn.CrossEntropyLoss(reduction='none')
    criterion2 = nn.CrossEntropyLoss()

    optimizer = optim.optim4GPU(cfg, model)
    trainer = train.Trainer(cfg, model, data_iter, optimizer, save_dir, get_device())

    writer = SummaryWriter(log_dir=log_dir) # for tensorboardX

    def get_loss(model, batch, global_step): # make sure loss is tensor
        input_ids, segment_ids, input_mask, masked_ids, masked_pos, masked_weights, is_next = batch

        logits_lm, logits_clsf = model(input_ids, segment_ids, input_mask, masked_pos)
        loss_lm = criterion1(logits_lm.transpose(1, 2), masked_ids) # for masked LM
        loss_lm = (loss_lm*masked_weights.float()).mean()
        loss_clsf = criterion2(logits_clsf, is_next) # for sentence classification
        writer.add_scalars('data/scalar_group',
                           {'loss_lm': loss_lm.item(),
                            'loss_clsf': loss_clsf.item(),
                            'loss_total': (loss_lm + loss_clsf).item(),
                            'lr': optimizer.get_lr()[0],
                           },
                           global_step)
        return loss_lm + loss_clsf

    trainer.train(get_loss, model_file, None, data_parallel)


# def classify(mode,
#         train_cfg='config/train_mrpc.json',
#          model_cfg='config/bert_base.json',
#          data_file='data/bert_pretrain.txt',
#          model_file=None,
#          pretrain_file='model/pretrain/tensorflow/bert_model.ckpt',
#          data_parallel=True,
#          vocab='data/vocab.txt',
#          save_dir='../exp/bert/mrpc',
#          max_len=64,
#          mode='train'):
#     if mode == 'pretrain':

    # cfg = train.Config.from_json(train_cfg)
    # model_cfg = models.Config.from_json(model_cfg)
    #
    # set_seeds(cfg.seed)
    #
    # tokenizer = tokenization.FullTokenizer(vocab_file=vocab, do_lower_case=True)
    #
    # TaskDataset = Binary # task dataset class according to the task
    # pipeline = [Tokenizing(tokenizer.convert_to_unicode, tokenizer.tokenize),
    #             AddSpecialTokensWithTruncation(max_len),
    #             TokenIndexing(tokenizer.convert_tokens_to_ids,
    #                           TaskDataset.labels, max_len)]
    #
    # dataset = TaskDataset(data_file, pipeline)
    # data_iter = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
    #
    # model = Classifier(model_cfg, len(TaskDataset.labels))
    # criterion = nn.CrossEntropyLoss()
    #
    # trainer = train.Trainer(cfg,
    #                         model,
    #                         data_iter,
    #                         optim.optim4GPU(cfg, model),
    #                         save_dir, get_device())
    #
    # if mode == 'train':
    #     def get_loss(model, batch, global_step): # make sure loss is a scalar tensor
    #         input_ids, segment_ids, input_mask, label_id = batch
    #         logits = model(input_ids, segment_ids, input_mask)
    #         loss = criterion(logits, label_id)
    #         return loss
    #
    #     trainer.train(get_loss, model_file, pretrain_file, data_parallel)

    # elif mode == 'eval':
    #     def evaluate(model, batch):
    #         input_ids, segment_ids, input_mask, label_id = batch
    #         logits = model(input_ids, segment_ids, input_mask)
    #         _, label_pred = logits.max(1)
    #         result = (label_pred == label_id).float() #.cpu().numpy()
    #         accuracy = result.mean()
    #         return accuracy, result
    #
    #     results = trainer.eval(evaluate, model_file, data_parallel)
    #     total_accuracy = torch.cat(results).mean().item()
    #     print('Accuracy:', total_accuracy)

if __name__ == '__main__':
    pretrain()