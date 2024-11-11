import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import cv2
import numpy as np
import torch.nn.init
import matplotlib.pyplot as plt
import skimage
from PIL import Image

class MyNet(nn.Module):
    def __init__(self,input_dim,num_conv):
        super(MyNet, self).__init__()
        self.num_conv = num_conv
        self.conv1 = nn.Conv2d(input_dim, 100, kernel_size=3, stride=1, padding=1 )
        self.bn1 = nn.BatchNorm2d(100)
        self.conv2 = nn.ModuleList()
        self.bn2 = nn.ModuleList()
        for i in range(self.num_conv-1):
            self.conv2.append( nn.Conv2d(100, 100, kernel_size=3, stride=1, padding=1 ) )
            self.bn2.append( nn.BatchNorm2d(100) )
        self.conv3 = nn.Conv2d(100, 100, kernel_size=1, stride=1, padding=0 )
        self.bn3 = nn.BatchNorm2d(100)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu( x )
        x = self.bn1(x)
        for i in range(self.num_conv-1):
            x = self.conv2[i](x)
            x = F.relu( x )
            x = self.bn2[i](x)
        x = self.conv3(x)
        x = self.bn3(x)
        return x

class UnsupervisedSegmentation():
    def __init__(self, img_full, n_iterations=25, lr=0.2, n_conv=2):
        self.img_full = img_full
        self.n_iterations = n_iterations
        self.lr = lr
        self.n_conv = n_conv

        self.use_cuda = torch.cuda.is_available()
    
    def prepare_slice(self, slice_idx):
        self.img = Image.fromarray(self.img_full[slice_idx, :, :])
        self.img = cv2.cvtColor(np.array(self.img), cv2.COLOR_RGB2BGR)

        self.data = torch.from_numpy( np.array([self.img.transpose( (2, 0, 1) ).astype('float32')/255.]) )

        if self.use_cuda:
            self.data = self.data.cuda()
        self.data = Variable(self.data)

    def train(self):
        # train
        self.model = MyNet( self.data.size(1), self.n_conv )
        if self.use_cuda:
            self.model.cuda()
        self.model.train()
        
        # similarity loss definition
        loss_fn = torch.nn.CrossEntropyLoss()
        
        # continuity loss definition
        loss_hpy = torch.nn.L1Loss(size_average = True)
        loss_hpz = torch.nn.L1Loss(size_average = True)
        
        HPy_target = torch.zeros(self.img.shape[0]-1, self.img.shape[1], 100)
        HPz_target = torch.zeros(self.img.shape[0], self.img.shape[1]-1, 100)
        if self.use_cuda:
            HPy_target = HPy_target.cuda()
            HPz_target = HPz_target.cuda()
        
        optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)

        for batch_idx in range(self.n_iterations):
            optimizer.zero_grad()
            output = self.model( self.data )[ 0 ]
            output = output.permute( 1, 2, 0 ).contiguous().view( -1, 100 )
        
            outputHP = output.reshape( (self.img.shape[0], self.img.shape[1], 100) )
            HPy = outputHP[1:, :, :] - outputHP[0:-1, :, :]
            HPz = outputHP[:, 1:, :] - outputHP[:, 0:-1, :]
            lhpy = loss_hpy(HPy,HPy_target)
            lhpz = loss_hpz(HPz,HPz_target)
        
            ignore, target = torch.max( output, 1 )
            im_target = target.data.cpu().numpy()
            nLabels = len(np.unique(im_target))
        
            loss = loss_fn(output, target) + (lhpy + lhpz)
                
            loss.backward()
            optimizer.step()
        
            print(batch_idx+1, '/', self.n_iterations, ' | loss :', loss.item())

        output = self.model( self.data )[ 0 ]
        output = output.permute( 1, 2, 0 ).contiguous().view( -1, 100 )
        _, output = torch.max( output, 1 )
        output = output.data.cpu().numpy()
        output = output.reshape( (self.img.shape[0], self.img.shape[1]) ).astype( np.uint8 )
        nLabels = len(np.unique(output))

        values = np.unique(output)
        avg = np.array([np.mean(self.img[output==i]) for i in values])
        
        self.segment_clr = np.empty(nLabels)
        for i in range(nLabels):
            idx = np.argsort(avg)[i]
            self.segment_clr[idx] = i*255/(nLabels-1)

    def infer(self):
        output = self.model( self.data )[ 0 ]
        output = output.permute( 1, 2, 0 ).contiguous().view( -1, 100 )
        _, output = torch.max( output, 1 )
        output = output.data.cpu().numpy()
        output = output.reshape( (self.img.shape[0], self.img.shape[1]) ).astype( np.uint8 )

        values = np.unique(output)

        final_segmentation = np.zeros_like(output)
        for r in range(output.shape[0]):
            for c in range(output.shape[1]):
                idx = np.where(values == output[r][c])[0][0]

                if idx >= 0 and idx < len(self.segment_clr):
                    final_segmentation[r][c] = self.segment_clr[idx]
                else:
                    final_segmentation[r][c] = self.img[r][c][0]

        return final_segmentation