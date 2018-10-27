import mxnet as mx
import os
import shutil
import sys
from time import time
import copy
import random
import argparse
import gluonnlp
import pickle
import numpy as np
from mxnet.gluon import nn, rnn
from mxnet.gluon.loss import SoftmaxCrossEntropyLoss
import gluonnlp
from gluonnlp.data import SpacyTokenizer
from gluonnlp.data import count_tokens
from mxnet.gluon.data import ArrayDataset, SimpleDataset
import codecs
from gluonnlp.data import FixedBucketSampler
import gluonnlp.data.batchify as btf
from mxnet.gluon.block import Block
from mxnet.gluon.data import DataLoader
from mxnet import gluon
parser = argparse.ArgumentParser(description = 'char_RNN')

#todo: Parser
parser.add_argument('--dataset', type = str, default = 'data/poetry.txt', help = 'Dataset to use.')
parser.add_argument('--epochs', type = int, default = 1000, help = 'upper epoch limit')
parser.add_argument('--mode', type = str, default = 'train', help = 'Train/Validation/Test.')
parser.add_argument('--experiment_name', type = str, default = 'experiment_test', help = 'experiment name')
parser.add_argument('--hidden_size', type = int, default = 128, help = 'dimension of RNN hidden states')
parser.add_argument('--embedding_size', type = int, default = 128, help = 'dimension of word embedding')
parser.add_argument('--batch_size', type = int, default = 64, help = 'Batch Size')
parser.add_argument('--test_batch_size', type = int, default = 16, help = 'Test Batch Size')
parser.add_argument('--beam_size', type = int, default = 4, help = 'beam size for beam search decoding.')
parser.add_argument('--max_vocab', type = int, default = 3500, help = 'Size of vocabulary.')
parser.add_argument('--seq_len', type = int, default=50, help = 'Sequence Length')
parser.add_argument('--vocab_dir', type=str, default=None, help = 'vocab path')
parser.add_argument('--optimizer', type = str, default = 'adam', help = 'Optimization Algorithm')
parser.add_argument('--lr', type = float, default = 0.15, help = 'Learning rate')
parser.add_argument('--bucket_ratio', type = float, default = 0.0, help = 'bucket_ratio')
parser.add_argument('--num_buckets', type = int, default = 100, help = 'bucket number')
parser.add_argument('--gpu', type = int, default = 3, help = 'id of the gpu to use. Set it to empty means to use cpu.')
parser.add_argument('--clip', type = float, default = 2.0, help = 'gradient clipping')
parser.add_argument('--log_interval', type=int, default=10, metavar='N', help='report interval')
parser.add_argument('--save_dir', type=str, default='out_dir_test', help='directory path to save the final model and training log')
parser.add_argument('--lp_alpha', type=float, default=1.0, help='Alpha used in calculating the length penalty')
parser.add_argument('--lp_k', type=int, default=5, help='K used in calculating the length penalty')
parser.add_argument('--log_root', type=str, default='log', help='log root')
parser.add_argument('--save_interval', type=int, default=1000, help='save interval')
args = parser.parse_args()

print("Finished Import")
DIR = args.vocab_dir
DATA_PATH = args.dataset
model_path = os.path.join('model', args.experiment_name)
if os.path.exists(model_path):
    shutil.rmtree(model_path)
os.mkdir(model_path)
VOCAB_PATH = os.path.join(model_path, 'my_vocab')
#todo: Data preprocessing
class TrainValDataTransform(object):
    def __init__(self, vocab=None, seq_len = None, max_vocab=5000):
        self._max_vocab = max_vocab
        self.my_vocab = vocab
        self.seq_len = seq_len
    
    def __call__(self, dataset, vocab = None):
        if vocab:
            return self.trans(dataset, vocab)
        else:
            return self.trans(dataset)
    

    def get_embedding(self):
        return gluonnlp.embedding.create(self._embedding_type, self._source)


    def read_text(self, dataset):
        with open(dataset, 'r') as ft:
            lines = ft.readlines()
        return lines

    def build_vocab(self, lines):
        print("Building, vocabulary")
        #cnt = 0
        for line in lines:
            #print(line)
            #cnt += 1
            #if cnt == 100:
            #    raise Exception("Debug")
            vocab_counter = count_tokens(line)
        
        self.my_vocab = gluonnlp.Vocab(vocab_counter, max_size = self._max_vocab)

        with open(os.path.join(model_path, "my_vocab"), 'wb') as fv:
            pickle.dump(self.my_vocab, fv, -1)
        with open(os.path.join(model_path, "vocab_list"), 'w') as wv:
            for word, count in vocab_counter.most_common(self._max_vocab):
                #print(type(word), type(count))
                wv.write(word + ' ' + str(count) + '\n')
        #except:
        #    raise Exception("error")
        #return my_vocab
    
    def check_vocab(self, dataset, vocab=None):
        if vocab is None:
            if os.path.exists(VOCAB_PATH):
                with open(VOCAB_PATH, 'rb') as fv:
                    print(VOCAB_PATH)
                    self.my_vocab = pickle.load(fv, encoding='utf-8')
            else:
                #raise Exception("Please build your vocab first")
                lines = self.read_text(dataset)
                self.build_vocab(lines)
        else:
            self.my_vocab = vocab

        if self.my_vocab is None:
            raise Exception("My_vocab Error")

    def trans(self, dataset, vocab = None):
        print("Transforming... ")

        self.check_vocab(dataset, vocab)
        print("Loading text... ")
        lines = self.read_text(dataset)

        if self.my_vocab is None:
            raise Exception("My_vocab Error")

        arr = self.text_to_arr(lines)
        
        inputs, targets = self.sample_generator(arr, self.seq_len)
        
        inputs = np.array(inputs)
        targets = np.array(targets)
        
        print("inputs shape: {}".format(inputs.shape))
        print("targets shape: {}".format(targets.shape))

        data = ArrayDataset(inputs, targets)
        
        return data, self.my_vocab

        
    
    def text_to_arr(self, lines):
        arr = []
        for line in lines:
            for ch in line:
                arr.append(self.my_vocab.token_to_idx[ch])
        
        return np.array(arr)
    

    def batch_generator(arr, n_seqs, n_steps):
        pass
        arr = copy.copy(arr)
        batch_size = n_seqs * n_steps
        n_batches = int(len(arr) / batch_size)
        arr = arr[:batch_size * n_batches]
        arr = arr.reshape((n_seqs, -1))
        while True:
            np.random.shuffle(arr)
            for n in range(0, arr.shape[1], n_steps):
                x = arr[:, n: n + n_steps]
                y = np.zeros_like(x)
                y[:, :-1], y[:, -1] = x[:1, 1:], x[:, 0]
                yield (x, y)

    def sample_generator(self, arr, seq_len):
        arr = copy.copy(arr)
        inputs = []
        targets = []
        for i in range(0, arr.shape[0], seq_len):
            x = arr[i: i + seq_len]
            y = np.zeros_like(x)
            y[:-1], y[-1] = x[1:], x[0]
            inputs.append(x)
            targets.append(y)
        return inputs, targets
            

#todo: get model
class charRNN(Block):
    def __init__(self, hidden_size = 128, 
                        embed_size = 128,
                        vocab = None,
                        initializer = mx.init.Uniform(0.02), 
                        dropout = 0.0,
                        embedding = None,
                        prefix = 'charRNN', 
                        params = None):
        super(charRNN, self).__init__(prefix=prefix, params = params)
        self.hidden_size = hidden_size
        self.params_init = initializer

        if embedding:
            self._embed = embedding
        else:
            self._embed = nn.HybridSequential(prefix='_embed_')
            with self.name_scope():

                self._embed.add(nn.Embedding(input_dim=len(vocab), 
                                            output_dim=embed_size,
                                            weight_initializer=self.params_init))
                self._embed.add(nn.Dropout(rate=dropout))

        with self.name_scope():
            self.lstm = rnn.LSTM(self.hidden_size,
                                num_layers = 2,
                                #layout = 'NTC',
                                i2h_weight_initializer=self.params_init,
                                h2h_weight_initializer=self.params_init,
                                i2h_bias_initializer = 'zeros',
                                h2h_bias_initializer = 'zeros',
                                prefix = 'lstm')
         
    
    def __call__(self, inputs, initial_state = None):
        #print("__call__: ", inputs.shape)
        #return self.forward(inputs, initial_state)
        return super(charRNN, self).__call__(inputs, initial_state)

    def forward(self, inputs, begin_state=None):
        encoded = self._embed(inputs)
        lstm_outputs, final_state = self.lstm(encoded, begin_state)
        return [lstm_outputs, final_state]

    def begin_state(self, *args, **kwargs):
        return self.lstm.begin_state(*args, **kwargs)

if args.gpu is None:
    ctx = mx.cpu()
    print('use CPU')
else:
    ctx = mx.gpu(args.gpu)

transformer = TrainValDataTransform(max_vocab=args.max_vocab, seq_len=args.seq_len)
data, my_vocab = transformer(DATA_PATH)
train_data = SimpleDataset([(ele[0], ele[1]) for i, ele in enumerate(data)])
train_data_lengths = [(len(ele[0]), len(ele[1])) for i, ele in enumerate(data)]
model = charRNN(hidden_size = args.hidden_size, 
                embed_size = args.embedding_size, 
                vocab = my_vocab)

loss_function = SoftmaxCrossEntropyLoss()
loss_function.hybridize()
model.initialize(init=mx.init.Uniform(0.02), ctx=ctx)
model.hybridize()


#todo: run_train()
def run_train():
    print("trainer")
    
    trainer = gluon.Trainer(model.collect_params(), args.optimizer, {'learning_rate':args.lr})
    
    train_batchify_fn = btf.Tuple(btf.Pad(), btf.Pad())
    
    #test_batchify_fn = btf.Tuple()

    train_batch_sampler = FixedBucketSampler(lengths = train_data_lengths,
                                             batch_size = args.batch_size,
                                             num_buckets = args.num_buckets,
                                             #ratio = args.bucket_ratio,
                                             shuffle = False)
    print(train_batch_sampler.stats()) 
    train_data_loader = DataLoader(train_data,
                                    batch_sampler = train_batch_sampler,
                                    batchify_fn = train_batchify_fn,
                                    num_workers = 1)
    
    print("Finished Loading")
    t0 = time()
    t1 = time()
    #arr = transformer(arr, args.num_seqs, args.num_steps)
    t2 = time()
    #train_data_loader = batch_generator(arr)
    for epoch_id in range(args.epochs):
        save_path = os.path.join(model_path, 'valid_best.params')
        new_state = None
        for idx, (lstm_input, target) in enumerate(train_data_loader):
            if lstm_input.shape[0] != args.batch_size:
                print("skip", idx)
                continue
            t3 = time()
            if new_state is None:
                new_state = model.begin_state(func=mx.nd.zeros, 
                                              batch_size=lstm_input.shape[0], 
                                              ctx=ctx)
            lstm_input = lstm_input.reshape((lstm_input.shape[1], -1))
            lstm_input = lstm_input.as_in_context(ctx)
            target = target.as_in_context(ctx)
            with mx.autograd.record():
                lstm_outputs, new_state = model(lstm_input, initial_state=new_state)

                lstm_outputs = mx.ndarray.stack(*lstm_outputs, axis=1)
                #print(lstm_outputs.shape, target.shape)
                loss = loss_function(lstm_outputs, target)
                loss = loss.mean()

            loss.backward()
            #grads = [p.grad(ctx) for p in model.collect_params().values()]
            #gnorm = gluon.utils.clip_global_norm(grads, args.clip)
            trainer.step(1)
            batch_loss = loss.asscalar()

            if idx % args.log_interval == 0:
                print('Epoch {} Batch: {}/{}... '.format(epoch_id, idx+1, len(train_data_loader)),
                        'loss: {:.4f}... '.format(batch_loss),
                        '{:.4f} sec/batch'.format(time() - t3))

            if idx % args.save_interval == 0:
                print('Save parameters to {}'.format(save_path))
                model.save_params(save_path)




if __name__ == '__main__':
    print("Start Training...")
    if args.mode == 'train':
        run_train()
