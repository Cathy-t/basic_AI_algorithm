import sys
import os
import argparse
import time
import datetime
import logging
import numpy as np
import json

import torch
import torch.nn as nn

from model import make_model
from utils import set_logger, read_vocab, write_vocab, build_vocab, Tokenizer, clip_gradient, adjust_learning_rate
from dataloader import create_split_loaders
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction

cc = SmoothingFunction()
from label_smoothing import *
from utils import padding_idx, sos_idx, eos_idx, unk_idx, NoamOpt, SimpleLossCompute, beam_search_decode
from tqdm import tqdm

# indicate src and tgt language
# src, tgt = 'en', 'zh'
# src_input, tgt_input = 'zh', 'en'
src_input, tgt_input = 'en', 'zh'


class Arguments():
    def __init__(self, config):
        for key in config:
            setattr(self, key, config[key])


def save_checkpoint(state, cp_file):
    torch.save(state, cp_file)


def count_paras(encoder, decoder, logging=None):
    '''
    Count model parameters.
    '''
    nparas_enc = sum(p.numel() for p in encoder.parameters())
    nparas_dec = sum(p.numel() for p in decoder.parameters())
    nparas_sum = nparas_enc + nparas_dec
    if logging is None:
        print('#paras of my model: enc {}M  dec {}M total {}M'.format(nparas_enc / 1e6, nparas_dec / 1e6,
                                                                      nparas_sum / 1e6))
    else:
        logging.info('#paras of my model: enc {}M  dec {}M total {}M'.format(nparas_enc / 1e6, nparas_dec / 1e6,
                                                                             nparas_sum / 1e6))


def setup(args, clear=False):
    '''
    Build vocabs from train or train/val set.
    '''
    TRAIN_VOCAB_EN, TRAIN_VOCAB_ZH = args.TRAIN_VOCAB_EN, args.TRAIN_VOCAB_ZH
    if clear:  ## delete previous vocab
        for file in [TRAIN_VOCAB_EN, TRAIN_VOCAB_ZH]:
            if os.path.exists(file):
                os.remove(file)
    # Build English vocabs
    if not os.path.exists(TRAIN_VOCAB_EN):
        write_vocab(build_vocab(args.DATA_DIR, language='en'), TRAIN_VOCAB_EN)
    # build Chinese vocabs
    if not os.path.exists(TRAIN_VOCAB_ZH):
        write_vocab(build_vocab(args.DATA_DIR, language='zh'), TRAIN_VOCAB_ZH)

    # set up seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


def main(args):
    model_prefix = '{}_{}'.format(args.model_type, args.train_id)

    log_path = args.LOG_DIR + model_prefix + '/'
    checkpoint_path = args.CHK_DIR + model_prefix + '/'
    result_path = args.RESULT_DIR + model_prefix + '/'
    cp_file = checkpoint_path + "best_model.pth.tar"
    init_epoch = 0

    if not os.path.exists(log_path):
        os.makedirs(log_path)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    ## set up the logger
    set_logger(os.path.join(log_path, 'train.log'))

    ## save argparse parameters
    with open(log_path + 'args.yaml', 'w') as f:
        for k, v in args.__dict__.items():
            f.write('{}: {}\n'.format(k, v))

    logging.info('Training model: {}'.format(model_prefix))

    ## set up vocab txt
    setup(args, clear=True)
    print(args.__dict__)

    maps = {'en': args.TRAIN_VOCAB_EN, 'zh': args.TRAIN_VOCAB_ZH}
    vocab_src = read_vocab(maps[src_input])
    tok_src = Tokenizer(language=src_input, vocab=vocab_src, encoding_length=args.MAX_INPUT_LENGTH)
    vocab_tgt = read_vocab(maps[tgt_input])
    tok_tgt = Tokenizer(language=tgt_input, vocab=vocab_tgt, encoding_length=args.MAX_INPUT_LENGTH)
    logging.info('Vocab size src/tgt:{}/{}'.format(len(vocab_src), len(vocab_tgt)))

    ## Setup the training, validation, and testing dataloaders
    train_loader, val_loader, test_loader = create_split_loaders(args.DATA_DIR, (tok_src, tok_tgt), args.batch_size,
                                                                 args.MAX_VID_LENGTH, (src_input, tgt_input),
                                                                 num_workers=4, pin_memory=True)
    logging.info('train/val/test size: {}/{}/{}'.format(len(train_loader), len(val_loader), len(test_loader)))

    # vocab = vocab_src + vocab_tgt

    ## init model
    model = make_model(len(vocab_src), len(vocab_tgt), N=args.nb_blocks, d_model=args.d_model, d_ff=args.d_model * 4,
                       h=args.att_h, dropout=args.dropout)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.train()

    ## define loss
    criterion = LabelSmoothing(size=len(vocab_tgt), padding_idx=padding_idx, smoothing=args.smoothing)
    ## init optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)  # zh2en
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99), eps=1e-9)
    model_opt = NoamOpt(args.d_model, 1, args.warmup_steps, optimizer)

    ## track loss during training
    total_train_loss, total_val_loss = [], []
    best_val_bleu, best_epoch = 0, 0

    ## init time
    zero_time = time.time()

    # Begin training procedure
    earlystop_flag = True
    rising_count = 0

    if True:

        for epoch in range(init_epoch, args.epochs):
            ## train for one epoch
            start_time = time.time()
            train_loss = train(train_loader, model, SimpleLossCompute(model.generator, criterion, opt=model_opt), epoch)

            val_loss, sentbleu, corpbleu = validate(val_loader, model,
                                                    SimpleLossCompute(model.generator, criterion, opt=None), epoch)
            end_time = time.time()

            epoch_time = end_time - start_time
            total_time = end_time - zero_time

            logging.info('Total time used: %s Epoch %d time uesd: %s train loss: %.4f val loss: %.4f corpbleu: %.4f' % (
                str(datetime.timedelta(seconds=int(total_time))),
                epoch, str(datetime.timedelta(seconds=int(epoch_time))), train_loss, val_loss, corpbleu))

            if corpbleu > best_val_bleu:
                best_val_bleu = corpbleu
                save_checkpoint({'epoch': epoch, 'state_dict': model.state_dict(),
                                 'optimizer': model_opt.optimizer.state_dict()}, cp_file)
                best_epoch = epoch

            logging.info("Finished {0} epochs of training".format(epoch + 1))

            total_train_loss.append(train_loss)
            total_val_loss.append(val_loss)

            if earlystop_flag:
                if epoch - best_epoch >= 12:
                    break

        logging.info('Best corpus bleu score {:.4f} at epoch {}'.format(best_val_bleu, best_epoch))

        ### the best model is the last model saved in our implementation
        logging.info('************ Start eval... ************')
        eval(test_loader, model, cp_file, tok_tgt, nbest=1, result_path=result_path)

    else:
        ### the best model is the last model saved in our implementation
        logging.info('************ Start eval... ************')
        eval(test_loader, model, cp_file, tok_tgt, nbest=1, result_path=result_path)


def train(train_loader, model, loss_compute, epoch):
    '''
    Performs one epoch's training.
    '''
    model.train()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for src, src_mask, trg, trg_mask, _, _, _ in tqdm(train_loader, desc="epoch {}/{}".format(epoch + 1, args.epochs)):
        src, src_mask, trg, trg_mask = src.cuda(), src_mask.cuda(), trg.cuda(), trg_mask.cuda()

        ntokens = (trg[:, 1:] != padding_idx).data.sum()

        out = model(src, src_mask, trg[:, :-1], trg_mask)
        ntokens_query = (src != padding_idx).data.sum()

        loss, _, _ = loss_compute(out, trg[:, 1:], ntokens, None, src, ntokens_query)

        # Keep track of metrics
        total_loss += loss.item()
        total_tokens += ntokens
        tokens += ntokens

    return total_loss / total_tokens.float()


def validate(val_loader, model, loss_compute, epoch):
    '''
    Performs one epoch's validation.
    '''
    model.eval()  # eval mode (no dropout or batchnorm)

    references = list()  # references (true captions) for calculating corpus BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)

    total_tokens = 0
    total_loss = 0
    tokens = 0

    with torch.no_grad():
        # Batches
        for src, src_mask, trg, trg_mask, _, _, tgtrefs in tqdm(val_loader, desc="epoch {}".format(epoch + 1)):

            src, src_mask, trg, trg_mask = src.cuda(), src_mask.cuda(), trg.cuda(), trg_mask.cuda()
            # Forward prop.
            out = model(src, src_mask, trg[:, :-1], trg_mask)

            ntokens_query = (src != padding_idx).data.sum()
            ntokens = (trg[:, 1:] != padding_idx).data.sum()
            loss, scores, pred_lengths = loss_compute(out, trg[:, 1:], ntokens, None, src, ntokens_query, eval=True,
                                                      max_len=args.maxlen)

            scores_copy = scores.clone()

            # Hypotheses
            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][1:pred_lengths[j]])  # remove pads and idx-0

            preds = temp_preds
            hypotheses.extend(preds)  # preds= [1,2,3]

            tgtrefs = [list(map(int, i.split())) for i in tgtrefs]  # tgtrefs = [[1,2,3], [2,4,3], [1,4,5,]]

            for r in tgtrefs:
                references.append([r])

            assert len(references) == len(hypotheses)

            total_loss += loss.item()
            total_tokens += ntokens
            tokens += ntokens

        # Calculate metrics
        avg_loss = total_loss / total_tokens.float()
        corpbleu = corpus_bleu(references, hypotheses)
        sentbleu = 0
        # for i, (r, h) in enumerate(zip(references, hypotheses), 1):
        #     sentbleu += sentence_bleu(r, h, smoothing_function=cc.method7)
        # sentbleu /= i

    return avg_loss, sentbleu, corpbleu


def eval(test_loader, model, cp_file, tok_tgt, nbest=1, result_path=None):
    '''
    Testing the model
    '''
    ### the best model is the last model saved in our implementation
    epoch = torch.load(cp_file)['epoch']
    logging.info('Use epoch {0} as the best model for testing'.format(epoch))
    model.load_state_dict(torch.load(cp_file)['state_dict'])

    model.eval()  # eval mode (no dropout or batchnorm)

    hypotheses = list()  # hypotheses (predictions)
    references = list()

    # generate sentences
    # logging.info('-----------------------generate--------------------------')
    start_time = time.time()

    hypotheses_eval = list()
    ids = list()

    with torch.no_grad():
        # Batches
        for cnt, (src, src_mask, sent_id, trg_y, srcrefs, tgtrefs) in enumerate(test_loader, 1):
            src, src_mask = src.cuda(), src_mask.cuda()
            vid = sent_id[0]
            ids.append(vid)

            # Forward prop.
            pred_out, _ = beam_search_decode(model, src, src_mask, args.maxlen, start_symbol=sos_idx,
                                             unk_symbol=unk_idx, end_symbol=eos_idx,
                                             pad_symbol=padding_idx)

            for n in range(min(nbest, len(pred_out))):
                pred = pred_out[n]
                temp_preds = []
                for w in pred[0]:
                    if w == eos_idx:
                        break
                    temp_preds.append(w)
                if n == 0:
                    hypotheses_eval.append(temp_preds)
                    preds = tok_tgt.decode_sentence(temp_preds)
                    hypotheses.append(preds)

                    tgtrefs = [list(map(int, i.split())) for i in tgtrefs]

                    for r in tgtrefs:
                        references.append([r])

                    assert len(references) == len(hypotheses_eval)

            # ids.extend(sent_id)

        ## save to json for submission
        dc = dict(zip(ids, hypotheses))
        print(len(dc))
        print('========time consuming : {}========'.format(time.time() - start_time))

        if not os.path.exists(result_path):
            os.makedirs(result_path)
        with open(result_path + 'results.json', 'w') as fp:
            json.dump(dc, fp, indent=4, ensure_ascii=False)

        corpbleu = corpus_bleu(references, hypotheses_eval)
        # sentbleu = 0
        # for i, (r, h) in enumerate(zip(references, hypotheses_eval), 1):
        #     sentbleu += sentence_bleu(r, h, smoothing_function=cc.method7)
        # sentbleu /= i
        # logging.info('test_data: corpbleu:{}, sentbleu:{}.'.format(corpbleu, sentbleu))
        logging.info('test_data: corpbleu:{}.'.format(corpbleu))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VGMT')
    parser.add_argument('--config', type=str, default='./configs.yaml')
    args = parser.parse_args()
    with open(args.config, 'r') as fin:
        import yaml

        args = Arguments(yaml.load(fin))
    main(args)
