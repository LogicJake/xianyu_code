import torch
import torch.nn as nn
import torch.nn.functional as F


# 离散特征统一 embedding 层
class EmbeddingLayer(nn.Module):
    def __init__(self, sparse_features, embedding_dim):
        super(EmbeddingLayer, self).__init__()

        embedding_dict = nn.ModuleDict()

        for name, num_embeddings in sparse_features.items():
            embedding_dict[name] = nn.Embedding(num_embeddings, embedding_dim)

        for tensor in embedding_dict.values():
            nn.init.normal_(tensor.weight, mean=0, std=0.01)

        self.embedding_dict = embedding_dict

    def forward(self, id_list):
        emb_list = []
        for id, name in id_list:
            emb = self.embedding_dict[name](id.long())
            emb = emb.view((emb.shape[0], 1, emb.shape[1]))
            emb_list.append(emb)

        output = torch.cat(emb_list, 1)

        return output


class FM(nn.Module):
    """
      Input shape
        - 3D tensor with shape: ``(batch_size,field_size,embedding_size)``.
      Output shape
        - 2D tensor with shape: ``(batch_size, 1)``.
    """
    def __init__(self):
        super(FM, self).__init__()

    def forward(self, inputs):
        fm_input = inputs

        square_of_sum = torch.pow(torch.sum(fm_input, dim=1, keepdim=True), 2)
        sum_of_square = torch.sum(fm_input * fm_input, dim=1, keepdim=True)
        cross_term = square_of_sum - sum_of_square
        cross_term = 0.5 * torch.sum(cross_term, dim=2, keepdim=False)

        return cross_term


def activation_layer(act_name):

    if isinstance(act_name, str):
        if act_name.lower() == 'sigmoid':
            act_layer = nn.Sigmoid()

        elif act_name.lower() == 'relu':
            act_layer = nn.ReLU(inplace=True)

    elif issubclass(act_name, nn.Module):
        act_layer = act_name()
    else:
        raise NotImplementedError

    return act_layer


class DNN(nn.Module):
    """
      Input shape
        - nD tensor with shape: ``(batch_size, ..., input_dim)``. The most common situation would be a 2D input with shape ``(batch_size, input_dim)``.

      Output shape
        - nD tensor with shape: ``(batch_size, ..., hidden_size[-1])``. For instance, for a 2D input with shape ``(batch_size, input_dim)``, the output would have shape ``(batch_size, hidden_size[-1])``.
    """
    def __init__(self,
                 inputs_dim,
                 hidden_units,
                 activation='relu',
                 dropout_rate=0,
                 use_bn=False,
                 seed=1024):
        super(DNN, self).__init__()
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
        self.seed = seed
        self.use_bn = use_bn
        if len(hidden_units) == 0:
            raise ValueError("hidden_units is empty!!")
        hidden_units = [inputs_dim] + list(hidden_units)

        self.linears = nn.ModuleList([
            nn.Linear(hidden_units[i], hidden_units[i + 1])
            for i in range(len(hidden_units) - 1)
        ])

        if self.use_bn:
            self.bn = nn.ModuleList([
                nn.BatchNorm1d(hidden_units[i + 1])
                for i in range(len(hidden_units) - 1)
            ])

        self.activation_layers = nn.ModuleList([
            activation_layer(activation) for i in range(len(hidden_units) - 1)
        ])

    def forward(self, inputs):
        deep_input = inputs

        for i in range(len(self.linears)):

            fc = self.linears[i](deep_input)

            if self.use_bn:
                fc = self.bn[i](fc)

            fc = self.activation_layers[i](fc)

            fc = self.dropout(fc)
            deep_input = fc
        return deep_input


class DeepFM(nn.Module):
    def __init__(self, sparse_features, dense_features, embedding_dim):
        super(DeepFM, self).__init__()

        self.embedding_layers = EmbeddingLayer(sparse_features, embedding_dim)

        # DNN
        input_size = len(dense_features) + len(sparse_features) * embedding_dim
        dnn_hidden_units = [5, 2]
        self.dnn = DNN(input_size, dnn_hidden_units)
        self.dnn_linear = nn.Linear(dnn_hidden_units[-1], 1, bias=False)
        self.fm = FM()

    def forward(self, base_cd, level, sex, tag, dense_features, target):
        id_list = [[base_cd, 'BASE_CD'], [level, '用户等级'], [sex, '性别'],
                   [tag, '标签']]
        
        embeding_input = self.embedding_layers(id_list)

        dnn_input = torch.cat(
            [torch.flatten(embeding_input, start_dim=1), dense_features], 1)

        dnn_output = self.dnn(dnn_input)
        dnn_logit = self.dnn_linear(dnn_output)

        fm_input = embeding_input
        fm_logit = self.fm(fm_input)

        logit = dnn_logit + fm_logit
        pred = torch.sigmoid(logit)
        pred = pred.squeeze()

        loss = F.binary_cross_entropy(pred, target)

        return pred, loss
