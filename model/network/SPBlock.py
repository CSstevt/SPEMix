import torch.nn as nn
class SPBlock(nn.Module):
    def __init__(self,input_dim,num_classes):
        super(SPBlock, self).__init__()
        self.fc=nn.Linear(input_dim,num_classes)
        self.mlp_proj = nn.Sequential(*[
            nn.Linear(input_dim, input_dim),
            nn.ReLU(inplace=False),
            nn.Linear(input_dim, 14)
        ])
        self.mclhead = nn.Linear(14, 2 * num_classes)
        self.opclassifier = nn.Linear(14, num_classes + 1)
    def forward(self,x):
        result={}
        project=self.mlp_proj(x)
        out=self.fc(x)
        mcl=self.mclhead(project)
        openresult=self.opclassifier(project)
        result['open']=openresult
        result['mcl']=mcl
        result['out']=out
        return result


