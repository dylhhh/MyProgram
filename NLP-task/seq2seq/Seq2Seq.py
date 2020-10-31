import torch
import numpy as np
from torch import nn
import utils


class Encoder(nn.Module):
    def __init__(self, en_vocab_size, emb_dim, hid_dim, num_layers, dr):
        super().__init__()
        self.embedding = nn.Embedding(en_vocab_size, emb_dim)
        self.num_layers = num_layers

        self.encoder = nn.GRU(emb_dim, hid_dim, num_layers, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dr)

    def forward(self, x):
        # x:[batch size, sequence_len]
        embedding = self.embedding(x)
        embedding = self.dropout(embedding)
        # output : [batch size, sequence len, hid dim * directions]
        # hidden :  [num_layers * directions, batch size  , hid dim]
        output, hidden_state = self.encoder(embedding)
        return output, hidden_state


class Decoder(nn.Module):
    def __init__(self, cn_vocab_size, emb_dim, hid_dim, num_layers, dr):
        super().__init__()
        self.embedding = nn.Embedding(cn_vocab_size, emb_dim)
        self.cn_vocab_size = cn_vocab_size

        # 加Attention
        self.attention = Attention(hid_dim * 2)

        self.decoder = nn.GRU(emb_dim, hid_dim * 2, num_layers, dropout=dr, batch_first=True)
        self.dropout = nn.Dropout(dr)

        # torch.nn.Linear（in_features，out_features，bias = True ）
        # self.linear1 = nn.Linear(hid_dim*2, hid_dim)
        self.linear2 = nn.Linear(hid_dim * 2, hid_dim * 4)
        self.linear3 = nn.Linear(hid_dim * 4, hid_dim * 8)
        self.linear4 = nn.Linear(hid_dim * 8, cn_vocab_size)

    def forward(self, y, decoder_hidden_state, encoder_output):
        # y:[batch_size, vocab_size]
        # decoder_hidden_state:[ num_layers*directions(1), batch_size, hid_dim]
        # encoder_output : [batch size, sequence len, hid dim * directions]
        y = y.unsqueeze(1)  # x:[batch_size,1, vocab_size]
        embedding = self.embedding(y)  ## embedding = [batch size, 1, emb dim]
        embedding = self.dropout(embedding)
        attention = self.attention(encoder_output, decoder_hidden_state)
        output, hidden_state = self.decoder(embedding, decoder_hidden_state + attention)
        output = self.linear2(output.squeeze(1))
        output = self.linear3(output)
        output = self.linear4(output)
        # [batch_size, vocab_size]

        return output, hidden_state


class Attention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.hid_dim = hid_dim

    def forward(self, encoder_output, decoder_hidden_state):
        # encoder_output : [batch size, sequence len, hid dim * directions(2)]
        # decoder_hidden_state : [ num_layers*directions(1), batch_size, hid_dim]

        decoder_hidden_state = decoder_hidden_state.permute(1, 2, 0)
        # decoder_hidden_state : [ batch_size, hid_dim, num_layers*directions(1)]

        attention_weight = torch.matmul(encoder_output, decoder_hidden_state)
        # attention_weight : [batch_size, sequence len, num_layers]

        attention_weight = nn.functional.softmax(attention_weight, dim=1)
        attention_weight = attention_weight.permute(0, 2, 1)
        # attention_weight : [batch_size, num_layers, sequence len]

        attention = torch.matmul(attention_weight, encoder_output)  # attention : [batch_size, num_layers, hid_dim]
        attention = attention.transpose(0, 1).contiguous()  # attention : [num_layers, batch_size, hid_dim]

        return attention


class seq2seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        assert encoder.encoder.num_layers == decoder.decoder.num_layers, \
            "Encoder and decoder must have equal number of layers!"

    def forward(self, input, target):
        # input  = [batch size, input len, en_vocab size]
        # target = [batch size, target len, cn_vocab size]
        # teacher_forcing_ratio 是有多少機率使用正確答案來訓練
        batch_size = target.shape[0]
        target_len = target.shape[1]
        vocab_size = self.decoder.cn_vocab_size

        # 準備一個儲存空間來儲存輸出
        outputs = torch.zeros(batch_size, target_len, vocab_size).to(self.device)

        encoder_output, hid_state = self.encoder(input)
        # 因為 Encoder 是双向的RNN，所以需要將同一層兩個方向的 hidden state 接在一起
        # hidden = [num_layers * directions, batch size  , hid dim]  --> [num_layers, directions, batch size  , hid dim]
        hid_state = hid_state.view(self.encoder.num_layers, 2, batch_size, -1)
        hid_state = torch.cat((hid_state[:, -2, :, :], hid_state[:, -1, :, :]), dim=2)
        # [num_layers , batch size  , hid dim * 2]

        # 取的 <BOS> token
        input = target[:, 0]
        preds = []
        for t in range(1, target_len):
            output, hidden = self.decoder(input, hid_state, encoder_output)
            outputs[:, t] = output
            # 決定是否用正確答案來做訓練
            teacher_force = np.random.random() <= utils.schedule_sampling(t, target_len)
            # 取出機率最大的單詞
            top1 = output.argmax(1)
            # 如果是 teacher force 則用正解訓練，反之用自己預測的單詞做預測
            input = target[:, t] if teacher_force and t < target_len else top1
            preds.append(top1.unsqueeze(1))
            preds = torch.cat(preds, 1)
            return outputs, preds

    def inference(self, input, target):
        # input  = [batch size, input len, en_vocab size]
        # target = [batch size, target len, cn_vocab size]
        # teacher_forcing_ratio 是有多少機率使用正確答案來訓練
        batch_size = target.shape[0]
        target_len = target.shape[1]
        vocab_size = self.decoder.cn_vocab_size

        # 準備一個儲存空間來儲存輸出
        outputs = torch.zeros(batch_size, target_len, vocab_size).to(self.device)

        encoder_output, hid_state = self.encoder(input)
        # 因為 Encoder 是双向的RNN，所以需要將同一層兩個方向的 hidden state 接在一起
        # hidden =  [num_layers * directions, batch size  , hid dim]  --> [num_layers, directions, batch size  , hid dim]
        hid_state = hid_state.view(self.encoder.num_layers, 2, batch_size, -1)
        hid_state = torch.cat((hid_state[:, -2, :, :], hid_state[:, -1, :, :]),
                              dim=2)  # [num_layers , batch size  , hid dim * 2]

        # 取的 <BOS> token
        input = target[:, 0]
        preds = []
        for t in range(1, target_len):
            output, hidden = self.decoder(input, hid_state, encoder_output)
            outputs[:, t] = output

            # 取出機率最大的單詞
            top1 = output.argmax(1)
            # 如果是 teacher force 則用正解訓練，反之用自己預測的單詞做預測
            input = top1
            preds.append(top1.unsqueeze(1))
            preds = torch.cat(preds, 1)
            return outputs, preds


