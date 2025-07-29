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
from utils import set_logger, read_vocab, write_vocab, build_vocab, Tokenizer, padding_idx
from dataloader import create_split_loaders
from train import setup
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from utils import padding_idx, sos_idx, eos_idx, unk_idx, beam_search_decode

cc = SmoothingFunction()


class Arguments():
    def __init__(self, config):
        for key in config:
            setattr(self, key, config[key])


def main(args):
    model_prefix = '{}_{}'.format(args.model_type, args.train_id)

    checkpoint_path = args.CHK_DIR + model_prefix + '/'
    result_path = args.RESULT_DIR + model_prefix + '/'
    cp_file = checkpoint_path + "best_model.pth.tar"

    if not os.path.exists(checkpoint_path):
        sys.exit('No checkpoint_path found {}'.format(checkpoint_path))

    print('Loading model: {}'.format(model_prefix))

    # set up vocab txt
    setup(args, clear=False)
    print(args.__dict__)

    # indicate src and tgt language
    src, tgt = 'en', 'zh'

    maps = {'en': args.TRAIN_VOCAB_EN, 'zh': args.TRAIN_VOCAB_ZH}
    vocab_src = read_vocab(maps[src])
    tok_src = Tokenizer(language=src, vocab=vocab_src, encoding_length=args.MAX_INPUT_LENGTH)
    vocab_tgt = read_vocab(maps[tgt])
    tok_tgt = Tokenizer(language=tgt, vocab=vocab_tgt, encoding_length=args.MAX_INPUT_LENGTH)
    print('Vocab size src/tgt:{}/{}'.format(len(vocab_src), len(vocab_tgt)))

    # Setup the training, validation, and testing dataloaders
    train_loader, val_loader, test_loader = create_split_loaders(args.DATA_DIR, (tok_src, tok_tgt), args.batch_size,
                                                                 args.MAX_VID_LENGTH, (src, tgt), num_workers=4,
                                                                 pin_memory=True)
    print('train/val/test size: {}/{}/{}'.format(len(train_loader), len(val_loader), len(test_loader)))

    ## init model
    model = make_model(len(vocab_src), len(vocab_tgt), N=args.nb_blocks, d_model=args.d_model, d_ff=args.d_model * 4,
                       h=args.att_h, dropout=args.dropout)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    ### load best model and eval
    print('************ Start eval... ************')
    eval(test_loader, model, cp_file, tok_tgt, nbest=1, result_path=result_path)


def eval(test_loader, model, cp_file, tok_tgt, nbest=1, result_path=None):
    '''
    Testing the model
    '''
    ### the best model is the last model saved in our implementation
    epoch = torch.load(cp_file)['epoch']
    print('Use epoch {0} as the best model for testing'.format(epoch))
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
        logging.info('test_data: corpbleu:{}.'.format(corpbleu))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VMT')
    parser.add_argument('--config', type=str, default='./configs.yaml')
    args = parser.parse_args()
    with open(args.config, 'r') as fin:
        import yaml

        args = Arguments(yaml.load(fin))
    main(args)
