import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
FUSION_MODEL=r"/opt/FSCE/resnet.pth"


class WeightedAttentionFusion(nn.Module):
    def __init__(self, input_dim):
        super(WeightedAttentionFusion, self).__init__()
        
        
        
        self.attn_weight_box = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.attn_weight_fused = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        
        self.fc = nn.Linear(input_dim * 2, input_dim)  

    def forward(self, box_features, fused_features):
        
        
        
        
        
        
        
        
        weighted_box = self.attn_weight_box * box_features
        weighted_fused = self.attn_weight_fused * fused_features
        
        
        combined_features = torch.cat((weighted_box, weighted_fused), dim=1) 
        
        
        batch_size, channels, height, width = combined_features.shape
        combined_features_flat = combined_features.view(batch_size, -1, height * width).permute(0, 2, 1)
        fused_output = self.fc(combined_features_flat).permute(0, 2, 1).view(batch_size, -1, height, width)
        
        
        output = box_features + fused_output
        
        return output










        



class TSFigureBackBone(nn.Module):
    def __init__(self, in_channels_list, out_channels=256):
        """
        初始化时空轨迹特征提取骨干网络。
        参数:
        - in_channels_list: 每个特征图的输入通道数列表，例如 [64, 256, 512, 1024]。
        - out_channels: 每个特征图经过卷积后的输出通道数。
        """
        super(TSFigureBackBone, self).__init__()
        
        
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1) for in_channels in in_channels_list
        ])
        
        
        self.fusion = WeightedAttentionFusion(out_channels)
        

    def forward(self, features, src_feat=None):
        """
        前向传播函数，对多层次特征图进行通道调整和插值操作。
        
        参数：
        - features: 一个列表，包含多层次特征图。128, 128]，64, 64]，
        - src_feat: 额外的输入特征，可与多层次特征融合。

        返回：
        - 融合后的特征图，形状为 [batch_size, out_channels, ...]。
        """
        if src_feat is None:
            raise ValueError("src_feat None")
        
        out = {}
        
        
        for i, (conv, feat, src) in enumerate(zip(self.convs, features, src_feat)):
            
            
            feat_out = conv(feat)
            feat_out = F.interpolate(feat_out, size=src_feat[src].shape[2:], mode='bilinear', align_corners=False)
            
            
            
                
                
            feat = self.fusion(fused_features = feat_out, box_features = src_feat[src])
            name = f"p{(i+2)}"
            out[name] = feat
            
            
            
            
            
            

        return out


class Fusion(nn.Module):
      def __init__(self,st_fig_channal_list=[64, 256, 512, 1024,2048], out_channels=256) -> None:
          super(Fusion,self).__init__()
          self.model = torch.load(FUSION_MODEL).eval()
          self.model = self.model.cuda()
          self.fusionNet = TSFigureBackBone(st_fig_channal_list, out_channels=out_channels).cuda()
      def forward(self,st_fig,src):
          features = self.model(st_fig)
          out = self.fusionNet(features,src)
          return out


import torch
from PIL import Image
from torchvision import transforms

def images_to_tensor(image_paths):
    i = 0  
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  
        transforms.ToTensor()           
    ])
    
    tensors = []
    for image_path in image_paths:
        
        image = Image.open(image_path).convert("RGB")   
        
        image_tensor = transform(image)
        tensors.append(image_tensor)
    stacked_tensors = torch.stack(tensors)
    if i == 0:
        stacked_tensors = stacked_tensors
    else:
        stacked_tensors = stacked_tensors.repeat(1,3,1,1)      
    
    return stacked_tensors

TEST_FUSION_TENSOR = None
'''



transform = transforms.Compose([
    transforms.Resize((256, 256)),  
    transforms.ToTensor()           
])

tensors = []
for image_path in image_paths:
    

        image = Image.open(image_path).convert("RGB")
        
        image_tensor = transform(image)
        tensors.append(image_tensor)
    else:
        image = Image.open(image_path).convert("L")
        
        image_tensor = transform(image)
        tensors.append(image_tensor)        
stacked_tensors = torch.stack(tensors)
if i == 0:
      stacked_tensors = stacked_tensors
else:
      stacked_tensors = stacked_tensors.repeat(1,3,1,1)
return stacked_tensors










model = Fusion(src=features)
model(test)
'''