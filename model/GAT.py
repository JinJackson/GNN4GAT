import torch
import torch.nn as nn
from utils.constants import LayerType

class GAT(torch.nn.Module):

    def __init__(self, num_of_layers, num_heads_per_layer, num_features_per_layer, add_skip_connection=True, bias=True,
                 dropout=0.6, layer_type=LayerType.IMP3, log_attention_weights=False):
        super().__init__()
        assert num_of_layers == len(num_heads_per_layer) == len(num_features_per_layer) - 1, f'Enter valid arch params.'

        GATLayer = get_layer_type(layer_type)   # fetch one of 3 available implementations
        num_heads_per_layer = [1] + num_heads_per_layer  # trick - so that I can nicely create GAT layers below

        get_layers = []  #collect GAT layers

        #TODO  未完成
        for i in range(num_of_layers):
            layer = GATLayer()


class GATLayer(nn.Module):
    # Base class for all implementations as there is much code that would otherwise be copy/pasted.

    # head所在维度
    head_dim = 1

    def __init__(self, num_in_features, num_out_features, num_of_heads, layer_type, concat=True, activation=nn.ELU(),
                 dropout_prob=0.6, add_skip_connection=True, bias=True, log_attention_weights=False):
        super().__init__()

        # Saving these as we'll need them in forward propagation in children layers (imp1/2/3)
        self.num_of_heads = num_of_heads
        self.num_out_features = num_out_features
        self.concat = concat  # Whether we need to concatenate or average the attention heads
        self.add_skip_connection = add_skip_connection

        # Trainable weights: linear projection matrix (denoted as "W" in the paper), attention target/source
        # (denoted as "a" in the paper) and bias (not mentioned in the paper but present in the official GAT repo)

        if layer_type == LayerType.IMP1:
            # Experimenting with different options to see what is faster (tip: focus on 1 implementation at a time)
            self.proj_param = nn.Parameter(torch.Tensor(num_of_heads, num_in_features, num_out_features))  #[heads, in_d, out_d]
        else:
            # You can treat this one matrix as num_of_heads independent W matrices
            self.linear_proj = nn.Linear(num_in_features, num_of_heads*num_out_features, bias=False)  #[in_d, heads*out_d]

        # After we concatenate target node (node i) and source node (node j) we apply the additive scoring function
        # which gives us un-normalized score "e". Here we split the "a" vector - but the semantics remain the same.

        # Basically instead of doing [x, y] (concatenation, x/y are node feature vectors) and dot product with "a"
        # we instead do a dot product between x and "a_left" and y and "a_right" and we sum them up
        # 这里的做法是把两个相邻的节点node i和node j分别对彼此做一次dot_product，然后再相加
        self.scoring_fn_target = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))
        self.scoring_fn_source = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))

        if layer_type == LayerType.IMP1:  #reshape for impl1
            self.scoring_fn_target = nn.Parameter(self.scoring_fn_target.reshape(num_of_heads, num_out_features, 1))
            self.scoring_fn_source = nn.Parameter(self.scoring_fn_source.reshape(num_of_heads, num_out_features, 1))

        # Bias is definitely not crucial to GAT - feel free to experiment (I pinged the main author, Petar, on this one)
        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(num_of_heads * num_out_features))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(num_out_features))
        else:
            self.register_parameter('bias', None)


        if add_skip_connection:
            self.skip_porj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)
        else:
            self.register_parameter('skip_proj', None)

        ###########End of Trainable weights###############

        self.leakyReLU == nn.LeakyReLU(0.2)  # using 0.2 as in the paper, no need to expose every setting
        self.softmax = nn.Softmax(dim=-1)    # -1 stands for apply the log-softmax along the last dimension
        self.activation = activation
        # Probably not the nicest design but I use the same module in 3 locations, before/after features projection
        # and for attention coefficients. Functionality-wise it's the same as using independent modules.
        
        self.dropout = nn.Dropout(p=dropout_prob)

        self.log_attention_weights = log_attention_weights  # whether we should log attention weights
        self.attention_weights = None

        self.init_params(layer_type)

    def init_params(self, layer_type):
        """
        The reason we're using Glorot (aka Xavier uniform) initialization is because it's a default TF initialization:
            https://stackoverflow.com/questions/37350131/what-is-the-default-variable-initializer-in-tensorflow

        The original repo was developed in TensorFlow (TF) and they used the default initialization.
        Feel free to experiment - there may be better initializations depending on your problem.
        """
        nn.init.xavier_uniform_(self.proj_param if layer_type == LayerType.IMP1 else self.linear_proj.weight)
        nn.init.xavier_uniform_(self.scoring_fn_target)
        nn.init.xavier_uniform_(self.scoring_fn_source)

        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def skip_concat_bias(self, attention_coefficients, in_nodes_features, out_nodes_features):
        if self.log_attention_weights:  # potentially log for later visualization in playground.py
            self.attention_weights = attention_coefficients

        #如果tensor的内存空间不连续，在进行view操作时会报错
        if not out_nodes_features.is_contiguous():
            out_nodes_features = out_nodes_features.contiguous()

        if self.add_skip_connection: # add skip or residual connection
            if out_nodes_features.shape[-1] == in_nodes_features.shape[-1]:  #in和out的维度相等
                # unsqueeze like this: 输入的向量(N, f_in) ->（N, 1, f_in), out_features输出向量(N, num_heads, f_out)
                # 所以需要对输入向量进行广播broadcast操作
                out_nodes_features += in_nodes_features.unsqueeze(1)
            else:
                #如果输入输出维度不相等，就是f_in != f_out，需要将f_in投影到可以与f_out相加的维度
                # feature vectors. skip_proj adds lots of additional capacity which may cause overfitting.
                out_nodes_features += self.skip_porj(in_nodes_features).view(-1, self.num_of_heads, self.num_out_features)

        if self.concat:
            # shape = (N, num_heads, f_out) -> (N, num_heads*f_out)
            out_nodes_features = out_nodes_features.view(-1, self.num_of_heads * self.num_out_features)
        else:
            #shape = (N, num_heads, f_out) -> (N, f_out), 这里直接取了均值？不相加吗
            out_nodes_features = out_nodes_features.mean(dim=self.head_dim)
        
        if self.bias is not None:
            out_nodes_features += self.bias

        return out_nodes_features if self.activation is None else self.activation(out_nodes_features)

def get_layer_type(layer_type):
    assert isinstance(layer_type, LayerType), f'Expected {LayerType} got {type(layer_type)}.'

    if layer_type == LayerType.IMP1:
        return GATLayerImp1
    elif layer_type == LayerType.IMP2:
        return GATLayerImp2
    elif layer_type == LayerType.IMP3:
        return GATLayerImp3
    else:
        raise Exception(f'Layer type {layer_type} not yet supported.')

class GATLayerImp2(GATLayer):
    """
        Implementation #2 was inspired by the official GAT implementation: https://github.com/PetarV-/GAT
        It's conceptually simpler than implementation #3 but computationally much less efficient.

        Note: this is the naive implementation not the sparse one and it's only suitable for a transductive setting.
        It would be fairly easy to make it work in the inductive setting as well but the purpose of this layer
        is more educational since it's way less efficient than implementation 3.
    """


    def __init__(self, num_in_features, num_out_features, num_of_heads, concat=True, activation=nn.ELU(),
                 dropout_prob=0.6, add_skip_connection=True, bias=True, log_attention_weights=False):
        super().__init__(num_in_features, num_out_features, num_of_heads, LayerType.IMP2, concat, activation, dropout_prob,
                         add_skip_connection, bias, log_attention_weights)

    def forward(self, data):
        #-----------Step1: Linear Projection + Regularization(use linear instead of matmul as in IMP1)-----------

        in_node_features, connectivity_mask = data   #unpack data
        num_of_nodes = in_node_features.shape[0]   #共有几个节点，即 N

        assert connectivity_mask.shape == (num_of_nodes, num_of_nodes), \
            f'Expected connectivity matrix with shape=({num_of_nodes},{num_of_nodes}), got shape={connectivity_mask.shape}.'

        # shape = (N, f_in)   N表示图中的节点， f_in表示结点的feature大小
        #我们对每个节点都进行dropout ----paper里是这样操作的
        in_node_features = self.dropout(in_node_features)

        # shape = (N, f_in) * (f_in, num_heads*f_out)  -> (N, num_heads, f_out)
        # 把f_in投影到num_heads个f_out上（每个头都分配f_out）
        nodes_features_proj = self.linear_proj(in_node_features).view(-1, self.num_of_heads, self.num_out_features)


        # -----------Setp2: Edge Attention calculation-----------
        # (using sum instead of bmm + additional permute calls -- compared to imp1)  bmm是两个矩阵的乘法

        # Apply scoring function (* represents element-wise(aka Hadamard) product) 按位乘
        # shape = (N, num_heads, f_out) * (1, num_heads, f_out)  --> (N, num_heads, 1)  求平均所以最后一维变成1
        # Optimization note: torch.sum() is as performant as .sum() in my experiments
        scores_source = torch.sum((nodes_features_proj * self.scoring_fn_source), dim=-1, keepdim=True)
        scores_target = torch.sum((nodes_features_proj * self.scoring_fn_target), dim=-1, keepdim=True)

        # src shape = (num_head, N, 1) and trg shape = (num_heads, 1, N)
        scores_source = scores_source.transpose(0, 1)
        scores_target = scores_target.permute(1, 2, 0)

        # shape=(num_heads, N, 1) + (num_heads, 1, N) -> (num_heads, N, N) 两个矩阵会自动broadcast
        # In Implementation 3 we are much smarter and don't have to calculate all NxN scores! (only E!)
        # Tip: it's conceptually easier to understand what happens here if you delete the NH dimension
        all_scores = self.leakyReLU(scores_source + scores_target)

        # connectivity mask will put -inf on all locations where there are no edges, after applying the softmax
        # this will result in attention scores being computed only for existing edges
        all_attention_coefficients = self.softmax(all_scores + connectivity_mask)



        # -----------Step3: Neighbour Aggregation (same as impl1)-----------
        # batch matrix multiply, shape = (num_heads, N, N) * (num_heads, N, f_out) -> (num_heads, N, f_out)
        out_nodes_features = torch.bmm(all_attention_coefficients, nodes_features_proj.transpose(0, 1))

        # Note: watch out here I made a silly mistake of using reshape instead of permute thinking it will
        # end up doing the same thing, but it didn't! The acc on Cora didn't go above 52%! (compared to reported ~82%)
        # shape = (N, NH, FOUT)
        out_nodes_features = out_nodes_features.permute(1, 0, 2)

        # Step 4: Residual/skip connections, concat and bias (same as in imp1)

        out_nodes_features = self.skip_concat_bias(all_attention_coefficients, in_node_features, out_nodes_features)
        return (out_nodes_features, connectivity_mask)

class GATLayerImp1(GATLayer):
    pass

class GATLayerImp3(GATLayer):
    pass