import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class GPHTBlock(nn.Module):
    def __init__(self, configs, depth):
        super().__init__()

        self.multipiler = configs.GT_pooling_rate[depth]
        self.patch_size = configs.token_len // self.multipiler
        self.down_sample = nn.MaxPool1d(self.multipiler)
        d_model = configs.GT_d_model
        d_ff = configs.GT_d_ff
        self.patch_embedding = PatchEmbedding(d_model, self.patch_size, self.patch_size, 0, configs.dropout)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(True, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), d_model,
                        configs.n_heads),
                    d_model,
                    d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                ) for l in range(configs.GT_e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.forecast_head = nn.Linear(d_model, configs.pred_len)

    def forward(self, x_enc):
        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1) #  b,nvar,len 32 7 96
        # u: [bs * nvars x patch_num x d_model]
        x_enc = self.down_sample(x_enc)  # 32 7 12  maxpooling
        enc_out, n_vars = self.patch_embedding.encode_patch(x_enc)
        enc_out = self.patch_embedding.pos_and_dropout(enc_out) # 224 2 512

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))

        bs = enc_out.shape[0]

        # Decoder
        dec_out = self.forecast_head(enc_out).reshape(bs, n_vars, -1)  # z: [bs x nvars x seq_len]
        return dec_out.permute(0, 2, 1)


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs

        self.encoders = nn.ModuleList([
            GPHTBlock(configs, i)
            for i in range(configs.depth)])

    def forecast(self, x_enc):
        means = x_enc.mean(1, keepdim=True)
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        seq_len = x_enc.shape[1]
        dec_out = 0

        for i, enc in enumerate(self.encoders):
            out_enc = enc(x_enc)
            dec_out += out_enc[:, -seq_len:, :]
            ar_roll = torch.zeros((x_enc.shape[0], self.configs.token_len, x_enc.shape[2])).to(x_enc.device) # 32 48 7
            ar_roll = torch.cat([ar_roll, out_enc], dim=1)[:, :-self.configs.token_len, :] # 32 96 7
            x_enc = x_enc - ar_roll

        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, dec_out.shape[1], 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, dec_out.shape[1], 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        return self.forecast(x_enc)


def get_config():
    import argparse
    import torch
    import random
    import numpy as np
    import os

    fix_seed = 2024
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='TimesNet')

    # basic config
    parser.add_argument('--task_name', type=str, required=False, default='long_term_forecast_GPHT',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=False, default=0, help='status')
    parser.add_argument('--model_id', type=str, required=False, default='test', help='model id')
    parser.add_argument('--model', type=str, required=False, default='GPHT',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')

    parser.add_argument('--GT_d_model', type=int, default=512)
    parser.add_argument('--GT_d_ff', type=int, default=2048)
    parser.add_argument('--token_len', type=int, default=48)
    parser.add_argument('--GT_pooling_rate', type=list, default=[8, 4, 2, 1])
    parser.add_argument('--GT_e_layers', type=int, default=3)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--load_pretrain', type=int, default=0, help='load pretrained model for further fine-tuning')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')

    # testing dataset for GPHT
    parser.add_argument('--transfer_data', type=str, required=False, default='ETTh1', help='dataset type')
    parser.add_argument('--transfer_root_path', type=str, default='dataset/ETT-small/',
                        help='root path of the data file')
    parser.add_argument('--transfer_data_path', type=str, default='ETTh1.csv', help='data file')

    parser.add_argument('--pretrain_batch_size', type=int, default=1024, help='batch size of pre-train input data')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--percent', type=int, default=100)

    # data loader
    parser.add_argument('--data', type=str, required=False, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='dataset/ETT-small/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')

    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')

    # settings of auto-regressive training for GPHT, and of training&testing for other models
    parser.add_argument('--seq_len', type=int, default=336, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=48, help='prediction sequence length')
    # settings of auto-regressive testing for GPHT
    parser.add_argument('--ar_seq_len', type=int, default=336)
    parser.add_argument('--ar_pred_len', type=int, default=96)

    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
    # model define
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')

    # optimization
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')


    args = parser.parse_args()

    return args


if __name__ == '__main__':
    configs = get_config()
    # configs.task_name='pretrain'
    configs.GT_d_model = 512

    configs.GT_d_ff=2048
    configs.token_len=48
    configs.GT_pooling_rate=[8,4,2,1]
    configs.GT_e_layers=3
    configs.depth=4
    configs.learning_rate=0.0001


    x= torch.randn(32,96,7)
    configs.devices = x.device
    model = Model(configs)
    c = model(x,0,0,0)
    d = 'end'