
    
#Import Stuff
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch.nn.functional as F
import random
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
import os
import functools
import pathlib
import inspect
import einops
from einops.layers.torch import Rearrange
from tqdm.auto import tqdm
import torchvision
output_dir = './Output/'
# Set device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Using {device} device")

#Hiperparameters 
num_samples = 1000000
batch_size = 1000
learning_rate = 1e-3
num_epochs = 100
T=300

print('Running with: ', 'num_samples = ', num_samples,'batch_size = ', batch_size, 'learning_rate= ',learning_rate,'num_epochs =',num_epochs,'T =',T )

#Create Training Data

#First create condition stuff (partons) = 4x 4 integer random martices 
clist =  [ torch.randint(low=400, high=1000, size=(4,4), dtype=torch.float32) for i in range(num_samples)]
#Define hadrons as square of the condition copied twice 
hadlist = [torch.cat((clist[i]@clist[i], clist[i]@clist[i], torch.zeros(56,4)), dim = 0) for i in range(num_samples) ] 
#Make a flattened list of conditions (format issues)
conditionlist = [torch.cat((clist[i], torch.zeros(60,4)), dim = 0) for i in range(num_samples) ] 
flatconditionlist = [conditionlist[i].flatten() for i in range(num_samples)]
#Convert lists to tensors of appropiate size
condition_tensor = torch.stack(flatconditionlist, dim=0)
hadron_tensor = torch.stack(hadlist, dim=0)
hadron_tensor = hadron_tensor.unsqueeze(1)


#Precompute Noise
noise_datalist = [torch.randn_like(torch.zeros((64, 4))) for i in range(num_samples)]
noise_data = torch.stack(noise_datalist, dim=0)

# Create Tensors
X = hadron_tensor.reshape(num_samples, 1, 16, 16)
y = noise_data.reshape(num_samples, 1, 16, 16)

#Reescale 
X_max = torch.max(X)
X = (1/X_max)*X
y_max = torch.max(y)
y = (1/y_max)*y
print(torch.max(X),torch.max(y), torch.min(X),torch.min(y))

#Put data in data set and load
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, X,y):
        super().__init__()
        self.data = X
        self.labels = y


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
#Create data set and load data
data_train, label_train = X, y 
train_dataset = MyDataset(data_train, label_train)
train_load = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
print('I created hadron and noise data sets of size: ', X.shape,y.shape)

##Forward Diffusion 
#Schedules 
def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2

def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start

# define beta schedule
betas = cosine_beta_schedule(timesteps=T)

# define alphas 
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)


# calculations for diffusion q(x_t | x_{t-1}) and others
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

# calculations for posterior q(x_{t-1} | x_t, x_0)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
# Forward diffusion (using the nice property)
def q_sample(x_start, t, noise):

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
#Diffusion example
shape = (2,1,16,16)
constant= torch.zeros(shape)
t = torch.tensor([299])
difuze = q_sample(constant,t,torch.rand((2,1,16,16)))
difuze.size()
#Visual
#array = difuze[0].squeeze().numpy()
# Plot the image using matplotlib
#plt.imshow(array, cmap='gray', vmin=0, vmax=1)
#plt.axis('off')
#plt.savefig(os.path.join(output_dir, 'ediffuzed.pdf'))
print('Forward difusion works well')
##UNET 
def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if inspect.isfunction(d) else d

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


class Residual(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim, dim_out=None):
    return torch.nn.Sequential(
        torch.nn.Upsample(scale_factor=2, mode="nearest"),
        torch.nn.Conv2d(dim, default(dim_out, dim), 3, padding=1),
    )


def Downsample(dim, dim_out=None):
    # No More Strided Convolutions or Pooling
    return torch.nn.Sequential(
        Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
        torch.nn.Conv2d(dim * 4, default(dim_out, dim), 1),
    )

class SinusoidalPositionEmbeddings(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class WeightStandardizedConv2d(torch.nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = einops.reduce(weight, "o ... -> o 1 1 1", "mean")
        var = einops.reduce(weight, "o ... -> o 1 1 1", functools.partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return torch.nn.functional.conv2d(
            x,
            normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

class Block(torch.nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm = torch.nn.GroupNorm(groups, dim_out)
        self.act = torch.nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(torch.nn.Module):
    """https://arxiv.org/abs/1512.03385"""

    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = (
            torch.nn.Sequential(torch.nn.SiLU(), torch.nn.Linear(time_emb_dim, dim_out * 2))
            if exists(time_emb_dim)
            else None
        )

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = torch.nn.Conv2d(dim, dim_out, 1) if dim != dim_out else torch.nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = einops.rearrange(time_emb, "b c -> b c 1 1")
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)

class Attention(torch.nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = torch.nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = torch.nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: einops.rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q * self.scale

        sim = torch.einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = torch.einsum("b h i j, b h d j -> b h i d", attn, v)
        out = einops.rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)

class LinearAttention(torch.nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = torch.nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = torch.nn.Sequential(torch.nn.Conv2d(hidden_dim, dim, 1), 
                                    torch.nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: einops.rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = einops.rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)

class PreNorm(torch.nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = torch.nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)
    
class Unet(torch.nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        self_condition=False,
        resnet_block_groups=4,
    ):
        super().__init__()

        # determine dimensions
        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = torch.nn.Conv2d(input_channels, init_dim, 1, padding=0) # changed to 1 and 0 from 7,3

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = functools.partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings
        time_dim = dim * 4

        self.time_mlp = torch.nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            torch.nn.Linear(dim, time_dim),
            torch.nn.GELU(),
            torch.nn.Linear(time_dim, time_dim),
        )

        # layers
        self.downs = torch.nn.ModuleList([])
        self.ups = torch.nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                torch.nn.ModuleList(
                    [
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Downsample(dim_in, dim_out)
                        if not is_last
                        else torch.nn.Conv2d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(
                torch.nn.ModuleList(
                    [
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Upsample(dim_out, dim_in)
                        if not is_last
                        else torch.nn.Conv2d(dim_out, dim_in, 3, padding=1),
                    ]
                )
            )

        self.out_dim = default(out_dim, channels)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = torch.nn.Conv2d(dim, self.out_dim, 1)

    def forward(self, x, time, x_self_cond=None):
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)


# Create an instance of the network
image_dim = 16
model = Unet(
    dim=image_dim,
    channels=1,
    dim_mults=(1, 2, 4,)
)

# Assuming you have defined and initialized your model as "model", move it to the GPU
model = model.to('cuda')
print('I created a U-Net')
# Defining the loss function and optimizer
loss_fn = torch.nn.MSELoss()
optimizer = Adam(model.parameters(), lr=learning_rate)
Loss = []
print('Begining training')
# Training the denoising model
for j in range(num_epochs):
    if j%10 ==0:
        print('Epoch: ', j)
    for batch, (X,y) in enumerate(train_load):
            T=300
            t = torch.randint(0, T, (batch_size,))
            x_t = q_sample(X,t,y).to(device)
            x_t = x_t.to('cuda')
            t = t.to('cuda')
            y = y.to('cuda')
            pred = model(x_t,t)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if batch % 100 == 0:
                print(f"  Loss (step {batch}):", loss.item())
            loss = loss.cpu()
            Loss.append(loss.detach().numpy().item())
#Plot loss
print('Training ended and loss lenght is: ', len(Loss))
l = [Loss[i] for i in range(len(Loss)) if i%100 ==0]
plt.plot(l)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig(os.path.join(output_dir, 'loss.pdf'))

# Sampling
@torch.no_grad()
def sample():
    x_T = torch.randn_like(torch.zeros((1,1,16,16)))
    print(x_T.shape)
    r =[T-i for i in range(0,T)]
    values = {}
    values[f"x_{T}"] = x_T
    for i in range(T):
        t = r[i]
        x_t = values[f"x_{t}"]
        values[f"x_{t-1}"] = p_sample(x_t, torch.tensor([t]))
    return values
def p_sample(x, t):
    t_index = t-torch.tensor([1])
    betas_t = extract(betas, t_index, x.shape)[0].item()
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t_index, x.shape)[0].item()
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t_index, x.shape)[0].item()
    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    x= x.to('cuda')
    t=t.to('cuda')
    pred_noise = model(x,t)
    model_mean = sqrt_recip_alphas_t * (x - betas_t/sqrt_one_minus_alphas_cumprod_t *pred_noise)
    if t == 1:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t_index, x.shape)[0].item()
        noise = torch.randn_like(x)
        return model_mean + np.sqrt(posterior_variance_t)*noise 
def get_sample():
    q = sample()
    sam = X_max*q['x_0']
    sam = torch.reshape(sam.squeeze(),(64,4))
    sam = sam.cpu()
    array = sam.numpy()
    return array
print('Sampling an example')
#Plot sample
s = get_sample()
# Plot the image using matplotlib
plt.imshow(s, cmap='gray', vmin=0, vmax=400)
plt.xlabel(' ')
plt.ylabel(' ')
plt.axis('on')
plt.savefig(os.path.join(output_dir, 'sample.pdf'))
