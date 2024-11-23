import math

import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding
import pywt
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
import numpy as np

from utils.masking import generate_causal_mask
from utils.model_utils import AdaLayerNorm, GELU2, Transpose


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
class TransformerEncoderBlock(nn.Module):
    def __init__(
        self, d_model: int, num_heads: int, feedforward_dim: int, dropout: float
    ):
        super(TransformerEncoderBlock, self).__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, d_model),
        )
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=feedforward_dim, kernel_size=1)
        self.activation = nn.GELU()
        self.conv2 = nn.Conv1d(in_channels=feedforward_dim, out_channels=d_model, kernel_size=1)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        """
        :param x: [batch_size * num_features, seq_len, d_model]
        :param mask: [1, 1, seq_len, seq_len]
        :return: [batch_size * num_features, seq_len, d_model]
        """
        # Self-attention
        attn_output, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-forward network
        ff_output = self.ff(x)
        output = self.norm2(x + self.dropout(ff_output))

        return output


class CausalTransformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        feedforward_dim: int,
        dropout: float,
    ):
        super(CausalTransformer, self).__init__()

        self.layers = nn.ModuleList(
            [
                TransformerEncoderBlock(d_model, num_heads, feedforward_dim, dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, is_mask=True):
        # x: [batch_size * num_features, seq_len, d_model]
        seq_len = x.size(1)
        mask = generate_causal_mask(seq_len).to(x.device) if is_mask else None
        for layer in self.layers:
            x = layer(x, mask)

        x = self.norm(x)
        return x


class GPHTBlock(nn.Module):
    def __init__(self, configs, depth):
        super().__init__()
        self.configs = configs
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
        activate="GELU"
        act = nn.GELU() if activate == 'GELU' else GELU2()
        # n_channel = seq_len

        trend_dim = self.configs.seq_len // self.multipiler // self.patch_size
        self.trend = TrendBlock(in_dim=self.configs.seq_len, out_dim=self.configs.seq_len, in_feat=1, out_feat=1,
                                act=act)
        # self.decomp = MovingBlock(n_channel)
        self.seasonal = FourierLayer(d_model=self.configs.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(self.configs.d_model, self.configs.d_model),
            act,
            nn.Linear(self.configs.d_model, self.configs.d_model),
            nn.Dropout(0.1),
        )
        self.ln2 = nn.LayerNorm(self.configs.d_model)

        self.encoder = CausalTransformer(
            d_model=configs.d_model,
            num_heads=configs.n_heads,
            feedforward_dim=configs.d_ff,
            dropout=configs.dropout,
            num_layers=configs.e_layers,
        )

        self.forecast_head = nn.Linear(d_model, configs.seq_len)

    def forward(self, x_enc):
        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1) #  b,nvar,len 1024 1 336
        # u: [bs * nvars x patch_num x d_model]
        x_enc = self.down_sample(x_enc)  # 1024 1 42  maxpooling
        enc_out, n_vars = self.patch_embedding.encode_patch(x_enc) # 1024 7 512
        enc_out = self.patch_embedding.pos_and_dropout(enc_out) # 1024 7 512

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out = self.encoder(enc_out)

        bs = enc_out.shape[0]

        # Decoder
        dec_out = self.forecast_head(enc_out).reshape(bs, n_vars, -1)[:,:,-self.configs.seq_len:]  # z: [bs x nvars x seq_len]

        # ------------------------------add begin
        dec_out = dec_out.permute(0, 2, 1)
        trend = self.trend(dec_out)  # in-->b,len,dim
        # enc_out = enc_out.permute(0, 2, 1)
        season = self.seasonal(dec_out)  # in-->b,len,dim
        # season = season.permute(0, 2, 1)
        # ------------------------------add end
        m = torch.mean(dec_out, dim=1, keepdim=True)

        # z: [bs x nvars x patch_num x d_model]

        return dec_out-m, trend, season,m


class CrossAttention(nn.Module):
    def __init__(self,
                 n_embd,  # the embed dim
                 condition_embd,  # condition dim
                 n_head,  # the number of heads
                 attn_pdrop=0.1,  # attention dropout prob
                 resid_pdrop=0.1,  # residual attention dropout prob
                 ):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(condition_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(condition_embd, n_embd)

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x, encoder_output, mask=None):
        B, T, C = x.size()
        B, T_E, _ = encoder_output.size()
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(encoder_output).view(B, T_E, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(encoder_output).view(B, T_E, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # (B, nh, T, T)

        att = F.softmax(att, dim=-1)  # (B, nh, T, T)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side, (B, T, C)
        att = att.mean(dim=1, keepdim=False)  # (B, T, T)

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, att


class FourierLayer(nn.Module):
    """
    Model seasonality of time series using the inverse DFT.
    """

    def __init__(self, d_model, low_freq=1, factor=1):
        super().__init__()
        self.d_model = d_model
        self.factor = factor
        self.low_freq = low_freq

    def forward(self, x):
        """x: (b, t, d)"""
        b, t, d = x.shape
        x_freq = torch.fft.rfft(x, dim=1)

        if t % 2 == 0:
            x_freq = x_freq[:, self.low_freq:-1]  # 32 49 64--32 47 64
            f = torch.fft.rfftfreq(t)[self.low_freq:-1]
        else:
            x_freq = x_freq[:, self.low_freq:]
            f = torch.fft.rfftfreq(t)[self.low_freq:]

        x_freq, index_tuple = self.topk_freq(x_freq)
        f = repeat(f, 'f -> b f d', b=x_freq.size(0), d=x_freq.size(2)).to(x_freq.device)
        f = rearrange(f[index_tuple], 'b f d -> b f () d').to(x_freq.device)
        return self.extrapolate(x_freq, f, t)

    def extrapolate(self, x_freq, f, t):
        x_freq = torch.cat([x_freq, x_freq.conj()], dim=1)
        f = torch.cat([f, -f], dim=1)
        t = rearrange(torch.arange(t, dtype=torch.float),
                      't -> () () t ()').to(x_freq.device)

        amp = rearrange(x_freq.abs(), 'b f d -> b f () d')
        phase = rearrange(x_freq.angle(), 'b f d -> b f () d')
        x_time = amp * torch.cos(2 * math.pi * f * t + phase)
        return reduce(x_time, 'b f t d -> b t d', 'sum')

    def topk_freq(self, x_freq):
        length = x_freq.shape[1]
        top_k = int(self.factor * math.log(length))
        values, indices = torch.topk(x_freq.abs(), top_k, dim=1, largest=True, sorted=True)
        mesh_a, mesh_b = torch.meshgrid(torch.arange(x_freq.size(0)), torch.arange(x_freq.size(2)), indexing='ij')
        index_tuple = (mesh_a.unsqueeze(1), indices, mesh_b.unsqueeze(1))
        x_freq = x_freq[index_tuple]
        return x_freq, index_tuple


class SeasonBlock(nn.Module):
    """
    Model seasonality of time series using the Fourier series.
    """

    def __init__(self, in_dim, out_dim, factor=1):
        super(SeasonBlock, self).__init__()
        season_poly = factor * min(32, int(out_dim // 2))
        self.season = nn.Conv1d(in_channels=in_dim, out_channels=season_poly, kernel_size=1, padding=0)
        fourier_space = torch.arange(0, out_dim, 1) / out_dim
        p1, p2 = (season_poly // 2, season_poly // 2) if season_poly % 2 == 0 \
            else (season_poly // 2, season_poly // 2 + 1)
        s1 = torch.stack([torch.cos(2 * np.pi * p * fourier_space) for p in range(1, p1 + 1)], dim=0)
        s2 = torch.stack([torch.sin(2 * np.pi * p * fourier_space) for p in range(1, p2 + 1)], dim=0)
        self.poly_space = torch.cat([s1, s2])

    def forward(self, input):
        b, c, h = input.shape
        x = self.season(input)
        season_vals = torch.matmul(x.transpose(1, 2), self.poly_space.to(x.device))
        season_vals = season_vals.transpose(1, 2)
        return season_vals


class TrendBlock(nn.Module):
    """
    Model trend of time series using the polynomial regressor.
    """

    def __init__(self, in_dim, out_dim, in_feat, out_feat, act):
        super(TrendBlock, self).__init__()
        '''
        32 1 96
            in_dim: in_channel      1
            out_dim: out_channel      1
            in_feat: dim or seqlen      96  
            out_feat: dim or seqlen      96 
        '''

        trend_poly = 3
        self.trend = nn.Sequential(
            nn.Conv1d(in_channels=in_dim, out_channels=trend_poly, kernel_size=3, padding=1),
            act,
            # Transpose(shape=(1, 2)),
            # nn.Conv1d(in_feat, out_feat, 3, stride=1, padding=1)
        )

        self.trend2 = Transpose(shape=(1, 2))
        self.trend3 = nn.Conv1d(in_feat, out_feat, 3, stride=1, padding=1)


        lin_space = torch.arange(1, out_dim + 1, 1) / (out_dim + 1)
        self.poly_space = torch.stack([lin_space ** float(p + 1) for p in range(trend_poly)], dim=0)

    def forward(self, input):
        b, c, h = input.shape
        # x = self.trend(input).transpose(1, 2)
        x = self.trend(input)
        x = self.trend2(x)
        x = self.trend3(x).transpose(1, 2)
        trend_vals = torch.matmul(x.transpose(1, 2), self.poly_space.to(x.device))
        trend_vals = trend_vals.transpose(1, 2)
        return trend_vals



class FuySeasonTrendLayer(nn.Module):
    def __init__(self,configs,i,activate='GELU'):
        super().__init__()
        self.configs =configs
        self.i = i
        assert activate in ['GELU', 'GELU2']
        act = nn.GELU() if activate == 'GELU' else GELU2()
        d_model = configs.GT_d_model
        d_ff = configs.GT_d_ff
        self.multipiler = configs.GT_pooling_rate[i]
        self.patch_size = configs.seq_len // self.multipiler
        # self.patch_size = configs.token_len // self.multipiler

        self.patch_embedding = PatchEmbedding(d_model, self.patch_size, self.patch_size, 0, configs.dropout)

        # n_channel = seq_len
        self.trend = TrendBlock(in_dim=1, out_dim=1,in_feat=self.configs.seq_len, out_feat=self.configs.seq_len, act=act)
        # self.decomp = MovingBlock(n_channel)
        self.seasonal = FourierLayer(d_model=self.configs.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(self.configs.d_model, self.configs.d_model),
            act,
            nn.Linear(self.configs.d_model, self.configs.d_model),
            nn.Dropout(0.1),
        )
        self.ln2 = nn.LayerNorm(self.configs.d_model)

    def forward(self,x):
        x = x.permute(0, 2, 1) #  b,nvar,len 1024 1 336

        # enc_out, n_vars = self.patch_embedding.encode_patch(x) # 1024 7 512
        # enc_out = self.patch_embedding.pos_and_dropout(enc_out) # 1024 7 512
        enc_out = x
        # enc_out = enc_out.permute(0, 2, 1)
        trend = self.trend(enc_out)
        enc_out = enc_out.permute(0,2,1)
        season = self.seasonal(enc_out)
        season = season.permute(0,2,1)

        enc_out = enc_out.permute(0,2,1)

        enc_out = enc_out + self.mlp(self.ln2(enc_out))
        mean = torch.mean(enc_out, dim=1, keepdim=True)


        return enc_out-mean,season,trend,mean


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.encoders = nn.ModuleList([
            GPHTBlock(configs, i)
            for i in range(configs.depth)])
        n_feat = self.configs.dec_in
        kernel_size = None
        padding_size = None


        self.blocks = nn.Sequential(*[
            FuySeasonTrendLayer(configs=self.configs,i=i,activate='GELU')
            for i in range(configs.depth)
        ])

    def forecast(self, x_enc):
        means = x_enc.mean(1, keepdim=True)
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        seq_len = x_enc.shape[1]
        dec_out = 0
        # x_enc = self.__multi_scale_process_inputs(x_enc)

        # c = self.decoder(x,None,None)

# ---------------------------------------
        b, c, _ = x_enc.shape
        # att_weights = []
        mean = []
        season = torch.zeros((b, c, 1), device=x_enc.device)  # 32 96 64 000
        trend = torch.zeros((b, c, 1), device=x_enc.device)  # 32 96 7 000
        # for block_idx in range(len(self.blocks)):
        #     x_enc,residual_season,residual_trend,residual_mean = self.blocks[block_idx](x_enc)
        #     season += residual_season
        #     trend += residual_trend
        #     mean.append(residual_mean)
        # mean = torch.cat(mean, dim=1)
        #
        # res = self.inverse(x_enc)  # 32 96 64---32 96 7
        # res_m = torch.mean(res, dim=1, keepdim=True)
        # season_error = self.combine_s(season.transpose(1, 2)).transpose(1, 2) + res - res_m  # season + res - resmean
        # trend = self.combine_m(mean) + res_m + trend  # trand + res + resmean

        # -------------------------------


        for i, enc in enumerate(self.encoders):
            x_enc, residual_trend, residual_season,residual_mean = enc(x_enc)
            season += residual_season
            trend += residual_trend
            mean.append(residual_mean)
        mean = torch.cat(mean, dim=1)

        dec_out = trend + season

            # ar_roll = torch.zeros((x_enc.shape[0], self.configs.token_len, x_enc.shape[2])).to(x_enc.device) # 32 48 7
            # ar_roll = torch.cat([ar_roll, out_enc], dim=1)[:, :-self.configs.token_len, :] # 32 96 7
            # x_enc = x_enc - ar_roll

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
    parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
    parser.add_argument('--down_sampling_method', type=str, default=None,
                        help='down sampling method, only support avg, max, conv')
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
    configs.down_sampling_window=2
    configs.down_sampling_method='avg'


    x= torch.randn(1024,336,1)
    configs.devices = x.device
    model = Model(configs)
    c = model(x,0,0,0)
    d = 'end'