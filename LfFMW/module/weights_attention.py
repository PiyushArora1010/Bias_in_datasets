import torch
import torch.nn as nn
from entmax import sparsemax


_EPSILON = 1e-6

def _vector_norms(v:torch.Tensor)->torch.Tensor:
    """ Computes the vector norms
    Args:
        v: The vector from which there must be calculated the norms

    Returns:
            A tensor containing the norms of input vector v
    """

    squared_norms = torch.sum(v * v, dim=1, keepdim=True)
    return torch.sqrt(squared_norms + _EPSILON)

def _distance(x:torch.Tensor , y:torch.Tensor, type:str='cosine')->torch.Tensor:
        """ Compute distances (or other similarity scores) between
        two sets of samples. Adapted from https://github.com/oscarknagg/few-shot/blob/672de83a853cc2d5e9fe304dc100b4a735c10c15/few_shot/utils.py#L45

        Args:
            x (torch.Tensor):  A tensor of shape (a, b) where b is the embedding dimension. In our paper a=1
            y (torch.Tensor):  A tensor of shape (m, b) where b is the embedding dimension. In our paper m is the number of samples in support set.
            type (str, optional): Type of distance to use. Defaults to 'cosine'. Possible values: cosine, l2, dot

        Raises:
            NameError: if the name of similarity is unknown

        Returns:
            torch.Tensor: A vector contining the distance of each sample in the vector y from vector x
        """
        if type == 'cosine':
            x_norm = x / _vector_norms(x)
            y_norm = y / _vector_norms(y)
            d = 1 - torch.mm(x_norm,y_norm.transpose(0,1))
        elif type == 'l2':
            d = (
                x.unsqueeze(1).expand(x.shape[0], y.shape[0], -1) -
                y.unsqueeze(0).expand(x.shape[0], y.shape[0], -1)
        ).pow(2).sum(dim=2)
        elif type == 'dot':
            expanded_x = x.unsqueeze(1).expand(x.shape[0], y.shape[0], -1)
            expanded_y = y.unsqueeze(0).expand(x.shape[0], y.shape[0], -1)
            d = -(expanded_x * expanded_y).sum(dim=2)
        else:
            raise NameError('{} not recognized as valid distance. Acceptable values are:[\'cosine\',\'l2\',\'dot\']'.format(type))
        return d

class attention_module(nn.Module):
    def __init__(self, input_features, output_features):
        super(attention_module, self).__init__()
        self.Q = nn.Linear(input_features, output_features)
        self.K = nn.Linear(1, output_features)
        # self.V = nn.Linear(input_features, output_features)

    def forward(self,image_features, mem_features):
        q = self.Q(image_features)
        k = self.K(mem_features)
        # v = self.V(mem_features)

        attention = torch.matmul(q,torch.transpose(k, 0, 1))
        attention_scores = torch.softmax(attention, dim=1)
        attention = attention_scores * image_features
        return attention, attention_scores

class WeightsAttention(nn.Module):

    def __init__(self, encoder_output_dim:int, output_dim:int, M:int, B:int, head: torch.nn.Module=None, classifier:torch.nn.Module=None,distance:str='cosine'):
        """ Initialize a Memory Wrap layer

        Args:
            encoder_output_dim (int): Dimensions of the last layer of the encoder
            output_dim (int): Number of desired output units.
            head (torch.nn.Module, optional): Module to use as read head. Input dimensions must be equal to encoder_output_dim. Default: torch.nn.Linear.
            classifier (torch.nn.Module, optional): Module to use as classifier head. Input dimensions must be equal to output dimensions*2 of the read head. Default: Multi-layer perceptron of dimensions [encoder_output_dim*2,encoder_output_dim*4,output_dim]
        """
        super(WeightsAttention, self).__init__()

        self.distance_name = distance
        self.attention_block = attention_module(M,B)
        self.classifier = classifier or torch.nn.Sequential(
            torch.nn.Linear(encoder_output_dim*2, encoder_output_dim*4),
            torch.nn.ReLU(),
           torch.nn.Linear(encoder_output_dim*4, output_dim)

        )
    


    def forward(self, encoder_output:torch.Tensor, memory_set:torch.Tensor, loss_weights, reweight:bool = False, return_weights:bool=False)->torch.Tensor:
        """Forward call of MemoryWrap.
        Args:
            input: A tensor of dimensions [b,dim] where dim is the dimension required by the encoder
            memory_set: Memory set. A tensor of dimension [m,dim] where m is the number of examples in memory
            parsed_memory: a flag to indicate if the memory set is already parsed by the encoder. It is useful
            to reduce the testing time if you fix the memory or if you parse the whole training set.
        Returns:
            A tuple `(output, content-weight)` where `output`
            is the output tensor, `content_weights` is a tensor containing the
            read weights for sample in memory. If return_weights is False, then
            only `output` is returned.
        """

        dist = _distance(encoder_output,memory_set,self.distance_name)
        content_weights = sparsemax(-dist,dim=1)

        if reweight:
            content_weights, _ = self.attention_block(content_weights, loss_weights)

        memory_vector = torch.matmul(content_weights,memory_set)

        # classifcation
        final_input = torch.cat([encoder_output,memory_vector],1)
        output = self.classifier(final_input)
       
        if return_weights:
            return output, content_weights
        else: 
            return output



class BaselineMemory(nn.Module):

    def __init__(self, encoder_output_dim:int, output_dim:int,head: torch.nn.Module=None, classifier:torch.nn.Module=None,distance:str='cosine'):
        """ Initialize the layer opf the baseline that uses only the memory set to compute the output

        Args:
            encoder_output_dim (int): Dimensions of the last layer of the encoder
            output_dim (int): Number of desired output units.
            head (torch.nn.Module, optional): Module to use as read head. Input dimensions must be equal to encoder_output_dim. Default: torch.nn.Linear.
            classifier (torch.nn.Module, optional): Module to use as classifier head. Input dimensions must be equal to output dimensions of the read head. Default: Multi-layer perceptron of dimensions [encoder_output_dim,encoder_output_dim*2,output_dim]
        """
        super(BaselineMemory, self).__init__()

        # Red Head
        self.distance = distance
        self.classifier = classifier or torch.nn.Sequential(
            torch.nn.Linear(encoder_output_dim, encoder_output_dim*2),
            torch.nn.ReLU(),
           torch.nn.Linear(encoder_output_dim*2, output_dim)

        )
        
    def forward(self, encoder_output:torch.Tensor, memory_set:torch.Tensor, return_weights:bool=False)->torch.Tensor:
        """Forward call of MemoryWrap.
        Args:
            input: A tensor of dimensions [b,dim] where dim is the dimension required by the encoder
            memory_set: Memory set. A tensor of dimension [m,dim] where m is the number of examples in memory
            parsed_memory: a flag to indicate if the memory set is already parsed by the encoder
        Returns:
            A tuple `(output, content-weight)` where `output`
            is the output tensor, `content_weights` is a tensor containing the
            read weights for sample in memory. If return_weights is False, then
            only `output` is returned.
        """

        dist = _distance(encoder_output,memory_set,self.distance)
        content_weights = sparsemax(-dist,dim=1)
        memory_vector = torch.matmul(content_weights,memory_set)
        output = self.classifier(memory_vector)

        if return_weights:
            return output, content_weights
        else: 
            return output