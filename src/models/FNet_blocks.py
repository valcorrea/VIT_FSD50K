import torch
from functools import partial
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import einsum, nn

class FNetBasicFourierTransform(nn.Module):
    def __init__(self, config, mixDim=(1,2)):
        super().__init__()
        self._init_fourier_transform(config, mixDim)

    def _init_fourier_transform(self, config, mixDim):
        # Dim 1 is patch dimension
        # Dim 2 is embedding dimension
        self.fourier_transform = partial(torch.fft.fftn, dim=mixDim)

    def forward(self, hidden_states):
        # NOTE: We do not use torch.vmap as it is not integrated into PyTorch stable versions.
        # Interested users can modify the code to use vmap from the nightly versions, getting the vmap from here:
        # https://pytorch.org/docs/master/generated/torch.vmap.html. Note that fourier transform methods will need
        # change accordingly.

        outputs = self.fourier_transform(hidden_states).real
        return outputs


class FNetBasicOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.LayerNorm(input_tensor + hidden_states)
        return hidden_states


class FNetFourierTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = FNetBasicFourierTransform(config)
        self.output = FNetBasicOutput(config)

    def forward(self, hidden_states):
        self_outputs = self.self(hidden_states)
        fourier_output = self.output(self_outputs[0], hidden_states)
        outputs = (fourier_output,)
        return outputs

class FNetWindowed(nn.Module):
    def __init__(self, config, window_size):
        super().__init__()
        assert (config.num_patches % window_size) == 0, "Number of patches must be divisible by window size"
        self.window_size = window_size
        self.self = FNetBasicFourierTransform(config, mixDim=(1))
        #self.net = nn.Sequential(
        #    nn.Linear(config.hidden_size, config.intermediate_size),
        #    nn.GELU(),
        #    nn.Dropout(config.hidden_dropout_prob),
        #    nn.Linear(config.intermediate_size, config.hidden_size),
        #    nn.Dropout(config.hidden_dropout_prob),
        #)
        self.embedFNet = FNetBasicFourierTransform(config, mixDim=(2))

    def forward(self, hidden_states):
        #i = torch.randint(0, hidden_states.shape[1], (1,),)
        #print("Shape of thing passed to FNet5 is: ", *hidden_states.shape)
        #x = hidden_states[:,1:,:]
        x = hidden_states
        #b, n, d = x.shape
        #window_size = n // 2
        #hidden_states = hidden_states.view(b, window_size, 2, d)
        x = rearrange(x, "b (n w) d -> b w d n", w=self.window_size)
        #print("Shape of x passed to FNet5: ", *x.shape)
        self_outputs = self.self(x)
        #print("Shape of self_outputs: ", *self_outputs.shape)
        self_outputs = rearrange(self_outputs, "b w d n -> b (n w) d")
        #self_outputs = self.net(self_outputs)
        
        # Maybe add basicFnet accross embed dimension here
        self_outputs = self.embedFNet(self_outputs)

        #hidden_states[:,1:,:] = self_outputs
        #outputs = hidden_states
        outputs = self_outputs
        #print("Exiting FNetWindowed")
        return outputs

class FNetMultiHead(nn.Module):
    def __init__(self, config):
        """
        Initialises Attention module. Config params:
        :param hidden_size: transformer dimension
        :param heads: number of attention heads
        :param hidden_dropout_prob: attention output dropout
        """
        super().__init__()
        inner_dim = config.hidden_size * config.heads
        project_out = not config.heads == 1

        self.heads = config.heads
        # Hardcoded for now
        window_sizes = (100, 50, 25, 100, 10, 5, 50, 25)
        #window_sizes = (101,)
        self.FNetWindowedBlocks = nn.ModuleList([FNetWindowed(config, window_sizes[i]) for i in range(self.heads)])
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, config.hidden_size), nn.Dropout(config.hidden_dropout_prob))
            if project_out
            else nn.Identity()
        )
        #self.embedFNet = FNetBasicFourierTransform(config, mixDim=(2))
        #self.generator = torch.Generator()
        #self.generator.manual_seed(1370210911620412)

    def forward(self, x):
        """
        Forward method for Attention module
        :param x: input tensor
        :return: Attention module output
        """
        #print("Shape of thing passed to FNet4 is: ", *x.shape)
        b, n, d, h = *x.shape, self.heads
        #b, n, d, h = 512, 101, 192, 1
        out = torch.empty((b, h, n, d), device=x.device, dtype=x.dtype)
        #index = torch.randint(0, n+1, (1,), generator=self.generator)[0]
        #mask = torch.ones((b, n, d), device=x.device, dtype=int)
        #mask[:,index,:] = 0
        #input = torch.cat((x[:,:index,:], x[:,index+1:,:]), dim=1)
        for i, f in enumerate(self.FNetWindowedBlocks):
            #print("Going through head number: ", i)
            out[:, i, :, :] = f(x)
        # Do Multi-head FNet here
        out = rearrange(out, "b h n d -> b n (h d)")
        #out = torch.cat((out[:,:index,:], x[:,index,None,:], out[:,index:,:]), dim=1)
        #print("Shape of FNet output: ", *out.shape)
        out = self.to_out(out)
        #out = self.embedFNet(out)
        return out