import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import scipy.io
#import h5py
import sklearn.metrics
import torch.nn as nn
from scipy.ndimage import gaussian_filter
import operator
from functools import reduce
from functools import partial


dx = 0.01
dz = 0.01
ns_num = 20

# dx_kernel = torch.tensor([[0, 0, 0],
#                         [-1/2/dx, 0, 1/2/dx],
#                         [0, 0, 0]], dtype=torch.float, requires_grad=False).view(1, 1, 3, 3)

# dz_kernel = torch.tensor([[0, -1/2/dz, 0],
#                         [0, 0, 0],
#                         [0, 1/2/dz, 0]], dtype=torch.float, requires_grad=False).view(1, 1, 3, 3)


# def conv_dx(x, dx_kernel):
        
#     batch_size, size_ns,  size_z, size_x = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
#     dx_kernel = dx_kernel.repeat(1, 1, 1, 1).to(device)
#     y_dx = torch.zeros(batch_size, size_ns, size_z, size_x, device=device)
#     for isou in range(size_ns):
#         y_dx[:,isou:isou+1,:,:] = F.conv2d(x[:,isou:isou+1,:,:], dx_kernel, stride=1, padding=1,)    
#     return y_dx

# def conv_dz(x, dz_kernel):
        
#     batch_size, size_ns,  size_z, size_x = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
#     dz_kernel = dz_kernel.repeat(1, 1, 1, 1).to(device)
#     y_dz = torch.zeros(batch_size, size_ns, size_z, size_x, device=device)
#     for isou in range(size_ns):
#         y_dz[:,isou:isou+1,:,:] = F.conv2d(x[:,isou:isou+1,:,:], dz_kernel, stride=1, padding=1,)    
#     return y_dz

def make_dx_kernel(dx: float) -> torch.Tensor:
    k = torch.tensor(
        [[0,          0,          0],
         [-0.5/dx,    0,    0.5/dx],
         [0,          0,          0]],
        dtype=torch.float32,
    ).view(1, 1, 3, 3)
    k.requires_grad_(False)
    return k

def make_dz_kernel(dz: float) -> torch.Tensor:
    k = torch.tensor(
        [[0,   -0.5/dz, 0],
         [0,         0, 0],
         [0,    0.5/dz, 0]],
        dtype=torch.float32,
    ).view(1, 1, 3, 3)
    k.requires_grad_(False)
    return k

dx_kernel = make_dx_kernel(dx)
dz_kernel = make_dz_kernel(dz)

# -------------------------------------------------
# Vectorized, device/dtype-safe conv helpers
# x: (B, ns, H, W) -> returns same shape
# -------------------------------------------------
def conv_dx(x: torch.Tensor, dx_kernel: torch.Tensor) -> torch.Tensor:
    B, ns, H, W = x.shape
    k = dx_kernel.to(device=x.device, dtype=x.dtype)          # match input
    x_ = x.reshape(B * ns, 1, H, W)                           # (B*ns,1,H,W)
    y_ = F.conv2d(x_, k, stride=1, padding=1)                 # (B*ns,1,H,W)
    return y_.reshape(B, ns, H, W)

def conv_dz(x: torch.Tensor, dz_kernel: torch.Tensor) -> torch.Tensor:
    B, ns, H, W = x.shape
    k = dz_kernel.to(device=x.device, dtype=x.dtype)
    x_ = x.reshape(B * ns, 1, H, W)
    y_ = F.conv2d(x_, k, stride=1, padding=1)
    return y_.reshape(B, ns, H, W)



class SpectralConv3d_fast(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d_fast, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2))
        # print("self.weights1:",self.weights1.shape)
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2))

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        #x_ft = torch.rfft(x,signal_ndim=3, normalized=True, onesided=True)
        x_ft = torch.stack((torch.fft.fftn(x,dim=(-3,-2,-1), norm="forward").real, torch.fft.fftn(x,dim=(-3,-2,-1), norm="forward").imag),dim=-1)
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.in_channels, x.size(-3), x.size(-2), x.size(-1)//2 + 1, 2, device=x.device)       
        # print("out_ft:",out_ft.shape)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        #Return to physical space
        #x = torch.irfft(out_ft, 3, normalized=True, onesided=True, signal_sizes=(x.size(-3), x.size(-2), x.size(-1)))
        out_ft = torch.complex(out_ft[:,:,:,:,:,0],out_ft[:,:,:,:,:,1]).squeeze(0)
        x = torch.fft.ifftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)),dim=(-3,-2,-1), norm="forward").real

        return x

#Complex multiplication
def compl_mul3d(a, b):
    # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
    op = partial(torch.einsum, "bixyz,ioxyz->boxyz")
    return torch.stack([
        op(a[..., 0], b[..., 0]) - op(a[..., 1], b[..., 1]),
        op(a[..., 1], b[..., 0]) + op(a[..., 0], b[..., 1])
    ], dim=-1)

################################################################
###################### 3d fourier layers #######################
################################################################

class SimpleBlock3d(nn.Module):
    def __init__(self, modes1, modes2, modes3, width):
        super(SimpleBlock3d, self).__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.fc0 = nn.Linear(2, self.width)
        # input channel is 12: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t)

        self.conv0 = SpectralConv3d_fast(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv1 = SpectralConv3d_fast(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv2 = SpectralConv3d_fast(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv3 = SpectralConv3d_fast(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv4 = SpectralConv3d_fast(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv5 = SpectralConv3d_fast(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv6 = SpectralConv3d_fast(self.width, self.width, self.modes1, self.modes2, self.modes3)

        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        self.w4 = nn.Conv1d(self.width, self.width, 1)
        self.w5 = nn.Conv1d(self.width, self.width, 1)
        self.w6 = nn.Conv1d(self.width, self.width, 1)

        self.bn0 = torch.nn.BatchNorm3d(self.width)
        self.bn1 = torch.nn.BatchNorm3d(self.width)
        self.bn2 = torch.nn.BatchNorm3d(self.width)
        self.bn3 = torch.nn.BatchNorm3d(self.width)
        self.bn4 = torch.nn.BatchNorm3d(self.width)
        self.bn5 = torch.nn.BatchNorm3d(self.width)
        self.bn6 = torch.nn.BatchNorm3d(self.width)

        
        num_neuron1, num_neuron2, num_neuron3, num_neuron4 = 128, 64, 32, 16
        self.fc1 = nn.Linear(self.width, num_neuron1)
        self.fc2 = nn.Linear(num_neuron1, num_neuron2)
        self.fc3 = nn.Linear(num_neuron2, num_neuron3)
        self.fc4 = nn.Linear(num_neuron3, num_neuron2)
        self.fc5 = nn.Linear(num_neuron2, num_neuron1)
        # self.fc6 = nn.Linear(num_neuron2, num_neuron1)

        self.fc7 = nn.Linear(num_neuron1, 1)

    def forward(self, x, y):
        
        #batchsize, size_ns, size_z, size_x, size_c
        batchsize, size_ns, size_z, size_x, size_c = x.shape[0], x.shape[1], x.shape[2], x.shape[3], x.shape[4]
        
        T0 = x[:,:,:,:,1]
        T0_dx = conv_dx(T0, dx_kernel)## batchsize, size_ns, size_z, size_x
        T0_dz = conv_dz(T0, dz_kernel)## batchsize, size_ns, size_z, size_x
        
        slope = 0.1
        x = self.fc0(x) #batchsize, size_ns, size_z, size_x, size_c
        x = x.permute(0, 4, 1, 2, 3) #batchsize, size_c, size_ns, size_z, size_x
        x1 = self.conv0(x)
        x2 = self.w0(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_ns, size_z, size_x)
        x = self.bn0(x1 + x2)
        x = F.leaky_relu(x,slope)       
        
        x1 = self.conv1(x)
        x2 = self.w1(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_ns, size_z, size_x)
        x = self.bn1(x1 + x2)
        x = F.leaky_relu(x,slope)       
        
        x1 = self.conv2(x)
        x2 = self.w2(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_ns, size_z, size_x)
        x = self.bn2(x1 + x2)
        x = F.leaky_relu(x,slope)     
        
        x1 = self.conv3(x)
        x2 = self.w3(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_ns, size_z, size_x)
        x = self.bn3(x1 + x2)
        x = F.leaky_relu(x,slope)    
        
        # x1 = self.conv4(x)
        # x2 = self.w4(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_ns, size_z, size_x)
        # x = self.bn4(x1 + x2)
        # x = F.leaky_relu(x,slope)       
      
        
        x = x.permute(0, 2, 3, 4, 1)
        
        x = self.fc1(x)#batchsize, size_ns, size_z, size_x, size_c
        x = F.leaky_relu(x,slope)
        x = self.fc2(x)
        x = F.leaky_relu(x,slope)
        x = self.fc3(x)
        x = F.leaky_relu(x,slope)
        x = self.fc4(x)#batchsize, size_ns, size_z, size_x, size_c
        x = F.leaky_relu(x,slope)
        x = self.fc5(x)
        x = F.leaky_relu(x,slope)
        # x = self.fc6(x)
        # x = F.leaky_relu(x,slope)
        
        x = self.fc7(x).view(batchsize, size_ns, size_z, size_x) #batchsize, size_ns, size_z, size_x
        mask = torch.where(T0<0.01,T0,torch.ones(batchsize, size_ns, size_z, size_x,device=x.device))
        mask[:,:,0:2,:] = 0
        x = x*mask
        
        tau_dx = conv_dx(x, dx_kernel)
        tau_dz = conv_dz(x, dz_kernel)
        
        loss = (tau_dx + T0_dx)**2 + (tau_dz + T0_dz)**2 - y

        return x, loss

class Net3d(nn.Module):
    def __init__(self, modes, width):
        super(Net3d, self).__init__()
        
        self.conv1 = SimpleBlock3d(modes, modes, modes, width)

    def forward(self, x, y):
        x = self.conv1(x, y)
        return x

    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))

        return c  