from bert.pretrain import *
import fire


def pretrain(train_cfg='config/pretrain.json',
         model_cfg='config/bert_base.json',
         data_file='data/bert_pretrain.txt',
         model_file=None,
         data_parallel=False,
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

    def get_loss(model, batch, global_step):  # make sure loss is tensor
        input_ids, segment_ids, input_mask, masked_ids, masked_pos, masked_weights, is_next = batch

        logits_lm, logits_clsf = model(input_ids, segment_ids, input_mask, masked_pos)
        loss_lm = criterion1(logits_lm.transpose(1, 2), masked_ids)  # for masked LM
        loss_lm = (loss_lm * masked_weights.float()).mean()
        loss_clsf = criterion2(logits_clsf, is_next)  # for sentence classification
        writer.add_scalars('data/scalar_group',
                           {'loss_lm': loss_lm.item(),
                            'loss_clsf': loss_clsf.item(),
                            'loss_total': (loss_lm + loss_clsf).item(),
                            'lr': optimizer.get_lr()[0],
                            },
                           global_step)
        return loss_lm + loss_clsf
    trainer.train(get_loss, model_file, None, data_parallel)


if __name__ == '__main__':
    fire.Fire(pretrain)