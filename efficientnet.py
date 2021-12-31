from torch import random
from torch.hub import load_state_dict_from_url
from .utils import *

class EfficientNet(nn.Module):
    def __init__(self, block_args_list, global_params):
        self.block_args_list = block_args_list
        self.global_params = global_params

        # parameters for batch normalization
        batch_norm_momentum = 1 - self.global_params.batch_norm_momentum
        batch_norm_epsilon = self.global_params.batch_norm_epsilon

        # stem
        in_channels = 3
        out_channels = round_filters(32, self.global_params)
        self._conv_stem = Conv2dSamePadding(in_channels, out_channels, kernel_size = 3, stride = 2, bias = False, name = 'stem_conv')
        self._bn0 = nn.BatchNorm2d(num_features = out_channels, momentum = batch_norm_momentum,eps = batch_norm_epsilon, name = 'stem_batch_norm')
        self._swish = Swish(name = 'swish')

        # build blocks
        idx = 0
        self._blocks = nn.ModuleList([])
        for i, block_args in enumerate(self.block_args_list):
            # update block input and output channel sizes based on the depth multiplier
            block_args = block_args._replace(
                input_filters = round_filters(block_args.input_filters, self.global_params),
                output_filters = round_filters(block_args.output_filters, self.global_params),
                num_repeat = round_repeats(block_args.num_repeat, self.global_params)
            )                    
            # the first block need to take care of the stride and filter size increase
            self._blocks.append(MBConvBlock(block_args, self.global_params, idx = idx))
            idx += 1
            # a single block has may have more than one layers ioncluded
            if block_args.num_repeat > 1:
                block_args = block_args._replace(input_filters = block_args.output_filters, strides = 1)
            # handle the rest of the blocks
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock(block_args, self.global_params, idx = idx))
                idx += 1
        
        # head
        in_channels = block_args.output_filters
        out_channels = round_filters(1280, self.global_params)
        self._conv_head = Conv2dSamePadding(in_channels, out_channels, kernel_size = 1, bias = False, name = 'head_conv')
        self._bn1 = nn.BatchNorm2d(num_features = out_channels, momentum = batch_norm_momentum, eps = batch_norm_epsilon, name = 'head_batch_norm')

        # final linear layer
        self.dropout_rate = self.global_params.dropout_rate
        self._fc = nn.Linear(out_channels, self.global_params.num_classes)

    def forward(self, x):
        # stem
        x = self._conv_stem(x)
        x = self._bn0(x)
        x = self._swish(x)

        # blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self.global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= idx / len(self._blocks)
            x = block(x, drop_connect_rate)

        # head
        x = self._conv_head(x)
        x = self._bn1(x)
        x = self._swish(x)

        # pool and dropour
        x = F.adaptive_avg_pool2d(x,output_size = 1).squeeze(-1).squeeze(-1) 
        if self.dropout_rate > 0:
            x = F.dropout(x, p = self.dropour_rate, training = self.training)
        
        # fully-connected layer
        x = self._fc(x)
        return x
    
    @classmethod
    def from_name(cls, model_name, *, n_classes = 1000, pretrained = False):
        return _get_model_by_name(model_name, classes = n_classes, pretrained = pretrained)
    @classmethod
    def encoder(cls, model_name,*,pretrained = True):
        model = cls.from_name(model_name, pretrained = pretrained)
        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.name = model_name
                self.global_params = model.global_params
                self.stem_conv = model._conv_stem
                self.stem_batch_norm = model._bn0
                self.stem_swish = Swish(name = 'stem_swish')
                self.blocks = model._blocks
                self.head_conv = model._conv_head
                self.head_batch_norm = model._bn1
                self.head_swish = Swish(name = 'head_swish')
            
            def forward(self, x):
                # stem
                x = self.stem_conv(x)
                x = self.stem_batch_norm(x)
                x = self.stem_swish(x)

                # blocks
                for i, block in enumerate(self.blocks):
                    drop_connect_rate = self.global_params.drop_connect_rate
                    if drop_connect_rate:
                        drop_connect_rate *= i / len(self.blocks)
                    x = block(x, drop_connect_rate)
                
                # head
                x = self.head_conv(x)
                x = self.head_batch_norm(x)
                x = self.head_swish(x)
                return x
        return Encoder()
    
    @classmethod
    def custom_head(cls, model_name, *, n_classes = 1000, pretrained = False):
        if n_classes == 1000:
            return cls.from_name(model_name, n_classes = n_classes, pretrained = pretrained)
        else:
            class CustomHead(nn.Module):
                def __init__(self, out_channels):
                    super().__init__()
                    self.encoder = cls.encoder(model_name, pretrained = pretrained)
                    self.custom_head = custom_head(self.n_channels*2, out_channels)
                
                @property
                def n_channels(self):
                    n_channels_dict = {
                        'efficientnet-b0' : 1280, 'efficientnet-b1' : 1280, 'efficientnet-b2' : 1408,
                        'efficientnet-b3' : 1536, 'efficientnet-b4' : 1792, 'efficientnet-b5' : 2048,
                        'efficientnet-b6' : 2304, 'efficientnet-b7' : 2560,
                    }

                    return n_channels_dict[self.encoder.name]
                
                def forward(self, x):
                    x = self.encoder(x)
                    mp = nn.AdaptiveMaxPool2d(output_size = (1,1))(x)
                    ap = nn.AdaptivePool2d(output_size = (1,1))(x)
                    x = torch.cat([mp,ap], dim = 1)
                    x = self.custom_head(x)

                    return x
                
        return CustomHead(n_classes)

def get_model_by_name(model_name, classes = 1000, pretrained = False):
    block_args_list, global_params = get_efficientnet_params(model_name, override_params={'num_classes' : classes})
    model = EfficientNet(block_args_list, global_params)

    try:
        if pretrained:
            pretrained_state_dict = load_state_dict_from_url(IMAGENET_WEIGHTS[model_name])

            if classes != 1000:
                random_state_dict = model.state_dict()
                pretrained_state_dict['_fc.weight'] = random_state_dict['_fc.weight']
                pretrained_state_dict['_fc.bias'] = random['_fc.bias']

            model.load_state_dict(pretrained_state_dict)
    except KeyError as e:
        print(f"NOTE : Currently model {e} does not have pretrained weights")
        
    return model




































































