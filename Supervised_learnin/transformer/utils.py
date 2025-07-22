import os
import sys
import re
import string
import json
import time
from collections import Counter
import numpy as np
import logging

# padding, unknown word, end of sentence
base_vocab = ['<PAD>', '<UNK>', '<SOS>', '<EOS>']
padding_idx = base_vocab.index('<PAD>')
sos_idx = base_vocab.index('<SOS>')
eos_idx = base_vocab.index('<EOS>')
unk_idx = base_vocab.index('<UNK>')

import torch


def set_logger(log_path):
    """
    Set the logger to log info in terminal and file `log_path`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)

def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))



### Build vocabulary, encode sentences
class Tokenizer(object):
    ''' Class to tokenize and encode a sentence. '''
    SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)') # Split on any non-alphanumeric character

    def __init__(self, language, vocab=None, encoding_length=30):
        self.language = language
        self.encoding_length = encoding_length
        self.vocab = vocab
        self.word_to_index = {}
        if vocab:
            for i,word in enumerate(vocab):
                self.word_to_index[word] = i

    def split_sentence(self, sentence):
        if self.language=='en':
            return self.split_sentence_en(sentence)
        elif self.language=='zh':
            return self.split_sentence_zh(sentence)

    def split_sentence_en(self, sentence):
        ''' Break sentence into a list of words and punctuation -- English '''
        toks = []
        for word in [s.strip().lower() for s in self.SENTENCE_SPLIT_REGEX.split(sentence.strip()) if len(s.strip()) > 0]:
            # Break up any words containing punctuation only, e.g. '!?', unless it is multiple full stops e.g. '..'
            if all(c in string.punctuation for c in word) and not all(c in '.' for c in word):
                toks += list(word)
            else:
                toks.append(word)
        return toks

    def split_sentence_zh(self, sentence):
        ''' Break sentence into a list of characters -- Chinese '''
        toks = []
        for char in sentence.strip():
            toks.append(char)
        return toks

    def encode_sentence(self, sentence, flag='SRC'):
        if len(self.word_to_index) == 0:
            sys.exit('Tokenizer has no vocab')
        encoding = []
        for word in self.split_sentence(sentence): # reverse input sentences
            if word in self.word_to_index:
                encoding.append(self.word_to_index[word])
            else:
                encoding.append(self.word_to_index['<UNK>'])
        ## cut words first since <EOS> should always be included in the end.
        if len(encoding) > self.encoding_length-2:
            encoding = encoding[:self.encoding_length-2]
        ## add <SOS> and <EOS>
        encoding = [self.word_to_index['<SOS>'], *encoding, self.word_to_index['<EOS>']] 
        length = min(self.encoding_length, len(encoding))
        if len(encoding) < self.encoding_length:
            encoding += [self.word_to_index['<PAD>']] * (self.encoding_length-len(encoding))
        cap = torch.from_numpy(np.array(encoding[:self.encoding_length]))

        # 做一个mask
        if flag == 'TRG':
            cap_mask = self.make_std_mask(cap[:-1], self.word_to_index['<PAD>']).squeeze(0)
        else:
            cap_mask = (cap != self.word_to_index['<PAD>']).unsqueeze(-2)  # src

        return cap, cap_mask, length


    def encode_sentence_nopad_2str(self, sentence):
        '''Encode a sentence without <SOS> and padding  '''
        if len(self.word_to_index) == 0:
            sys.exit('Tokenizer has no vocab')
        encoding = []
        for word in self.split_sentence(sentence): # reverse input sentences
            if word in self.word_to_index:
                encoding.append(self.word_to_index[word])
            else:
                encoding.append(999999)

        string = ' '.join([str(i) for i in np.array(encoding)])
        return string # exclude <SOS>


    def decode_sentence(self, encoding):
        sentence = []
        for ix in encoding:
            if ix == self.word_to_index['<PAD>']:
                break
            else:
                if ix >= len(self.vocab):
                    sentence.append('<UNK>')
                else:
                    sentence.append(self.vocab[ix])
        return " ".join(sentence) # unreverse before output


    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
        return tgt_mask


def build_vocab(data_dir, language, min_count=5, start_vocab=base_vocab):
    ''' Build a vocab, starting with base vocab containing a few useful tokens. '''
    assert language in ['en', 'zh']
    count = Counter()
    t = Tokenizer(language)

    with open(data_dir+'vatex_train_data.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
    lan2cap={'en':'enCap', 'zh':'chCap'}
    for d in data:
        for cap in d[lan2cap[language]]:
            count.update(t.split_sentence(cap))
    vocab = list(start_vocab)
    for word,num in count.most_common():
        if num >= min_count:
            vocab.append(word)
        else:
            break
    return vocab


def write_vocab(vocab, path):
    print ('Writing vocab of size %d to %s' % (len(vocab),path))
    with open(path, 'w', encoding='utf-8') as f:
        for word in vocab:
            f.write("%s\n" % word)

def read_vocab(path):
    vocab = []
    with open(path, encoding='utf-8') as f:
        vocab = [word.strip() for word in f.readlines()]
    return vocab


class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))


class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, generator, criterion, ae_generator=None, opt=None, l=1.0):
        self.generator = generator
        self.ae_generator = ae_generator
        self.criterion = criterion
        self.opt = opt
        self.l = l

    def __call__(self, x, y, norm, ae_x=None, ae_y=None, ae_norm=None, eval=False, max_len=40):
        """greedy decoder for calculate sentbleu"""
        if eval:
            batch_size = x.size(0)
            pred_lengths = [0] * batch_size  # 表示batch_size中每个句子的实际长度
            scores = self.generator.inference(x)
            for i in range(batch_size):
                seq = scores[i]  # torch.Size([40, 7308])
                _, words = torch.max(seq, dim=1)  # words: torch.Size([40])
                for t in range(words.size(0)):
                    word = words[t]
                    if word == eos_idx and pred_lengths[i] == 0:
                        pred_lengths[i] = t
                if pred_lengths[i] == 0:
                    pred_lengths[i] = max_len
        else:
            scores, pred_lengths = None, None

        out = self.generator(x)
        loss = self.criterion(out.contiguous().view(-1, out.size(-1)),
                              y.contiguous().view(-1)) / norm.float()

        if self.opt is not None:   # train
            loss.backward()
            self.opt.step()
            self.opt.optimizer.zero_grad()

        return loss.item() * norm.float(), scores, pred_lengths


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def beam_search_decode(model, src, src_mask, max_len, start_symbol, unk_symbol, end_symbol, pad_symbol, beam=5, penalty=1.0,
                       nbest=5, min_len=1):
    query, query_mask = src, src_mask
    query_memory = model.encode(query, query_mask)

    ds = torch.ones(1, 1).fill_(start_symbol).type_as(query.data)
    hyplist = [([], 0., ds)]
    best_state = None
    comp_hyplist = []
    for l in range(max_len):
        new_hyplist = []
        argmin = 0
        for out, lp, st in hyplist:
            cap2res_mask = None
            output = model.decode(query_memory, query_mask, st,
                                  subsequent_mask(st.size(1)).type_as(query.data))
            if type(output) == tuple or type(output) == list:
                logp = model.generator(output[0][:, -1])
            else:
                logp = model.generator(output[:, -1])
            lp_vec = logp.cpu().data.numpy() + lp
            lp_vec = np.squeeze(lp_vec)
            if l >= min_len:
                new_lp = lp_vec[end_symbol] + penalty * (len(out) + 1)
                comp_hyplist.append((out, new_lp))
                if best_state is None or best_state < new_lp:
                    best_state = new_lp
            count = 1
            for o in np.argsort(lp_vec)[::-1]:
                if o == unk_symbol or o == end_symbol:
                    continue
                new_lp = lp_vec[o]
                if len(new_hyplist) == beam:
                    if new_hyplist[argmin][1] < new_lp:
                        new_st = torch.cat([st, torch.ones(1, 1).type_as(query.data).fill_(int(o))], dim=1)
                        new_hyplist[argmin] = (out + [o], new_lp, new_st)
                        argmin = min(enumerate(new_hyplist), key=lambda h: h[1][1])[0]
                    else:
                        break
                else:
                    new_st = torch.cat([st, torch.ones(1, 1).type_as(query.data).fill_(int(o))], dim=1)
                    new_hyplist.append((out + [o], new_lp, new_st))
                    if len(new_hyplist) == beam:
                        argmin = min(enumerate(new_hyplist), key=lambda h: h[1][1])[0]
                count += 1
        hyplist = new_hyplist

    if len(comp_hyplist) > 0:
        maxhyps = sorted(comp_hyplist, key=lambda h: -h[1])[:nbest]
        return maxhyps, best_state
    else:
        return [([], 0)], None
