import torch


# For second parameter of our model
bias_bool = True

#--------------------------------------------------------------------------#


# =========================================================================#
#                             Network Definitions
# =========================================================================#


# Network_1
#--------------------------------------------------------------------------#
class Net_one(torch.nn.Module):

    def __init__(self, dim, ns, act_func, hh, device):
        super().__init__()
        self.lin1 = torch.nn.Linear(dim+1, ns, bias=bias_bool)
        #self.lin2 = torch.nn.Linear(ns, ns, bias=bias_bool)
        #self.lin3 = torch.nn.Linear(ns, ns, bias=bias_bool)
        self.linlast = torch.nn.Linear(int(ns), 1)
        self.act_func = act_func
        

        self.lintt = torch.nn.Linear(1, dim)
        self.dim = dim
        self.hh = hh
        self.device = device

        
    def forward(self, t, inp):
       
        out = torch.cat((t, inp),dim=1)
        out = self.act_func(self.lin1(out))
        #out = self.act_func(out + self.hh * self.lin2(out))
        #out = self.act_func(out + self.hh * self.lin3(out))
        out = self.linlast(out)
        
        return out
#-------------------------------------------------------------------------#

# Network_2
#-------------------------------------------------------------------------#
class Net_two(torch.nn.Module):

    def __init__(self, dim, ns, act_func, hh, device):
        super().__init__()
        
        self.lin1 = torch.nn.Linear(dim + 1, ns)
        #self.lin2 = torch.nn.Linear(ns, ns)
        #self.lin3 = torch.nn.Linear(ns, ns)
        self.linlast = torch.nn.Linear(int(ns), 1)
        self.act_func = act_func

        self.lintt = torch.nn.Linear(1, dim, bias=True)
        self.dim = dim
        self.hh = hh
        self.device = device

    def forward(self, t, inp):
        
        
        out = torch.cat((t, inp),dim=1)

        out = self.act_func(self.lin1(out))
        #out = self.act_func(out + self.hh * self.lin2(out))
        #out = self.act_func(out + self.hh * self.lin3(out))
        out = self.linlast(out)

        
        # In the case of congestion return abs(out) 
        return out
#-------------------------------------------------------------------------#
#-------------------------------------------------------------------------#

#Control Network
#-------------------------------------------------------------------------#
class Control(torch.nn.Module):

    def __init__(self, dim, ns, act_func, hh, device):
        super().__init__()
        #self.mu = mu
        #self.std = std

        self.lin1 = torch.nn.Linear(dim + 1, ns)
        #self.lin2 = torch.nn.Linear(ns, ns)
        #self.lin3 = torch.nn.Linear(ns, ns)
        self.linlast = torch.nn.Linear(int(ns), 1)
        self.act_func = act_func

        self.lintt = torch.nn.Linear(1, dim, bias=True)
        self.dim = dim
        self.hh = hh
        self.device = device

    def forward(self, t, inp):
        
        
        out = torch.cat((t, inp),dim=1)

        out = self.act_func(self.lin1(out))
        #out = self.act_func(out + self.hh * self.lin2(out))
        #out = self.act_func(out + self.hh * self.lin3(out))
        out = self.linlast(out)
        
        return out
