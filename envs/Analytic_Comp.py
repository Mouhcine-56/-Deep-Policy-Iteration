import math
import numpy as np
import torch
from utils.utils import one_STRING, two_STRING, sqeuc
from sklearn.neighbors import KernelDensity

# =======================================
#           Analytic_Comparison
# =======================================
class AnalyticEnv(object):
    
    def __init__(self, device):
        self.dim = 1
        self.nu = 1
        self.TT = 1
        self.lam_congestion = 0. # without congestion 
        self.device = device
        self.name = "AnalyticEnv"  # Environment name

        # Parameters for the analytic example
        self.an_lam = self.lam_congestion
        self.an_nu = self.nu
        self.an_beta = 1
        self.an_dim = self.dim
        self.an_alpha = (-self.an_lam + np.sqrt(self.an_lam * self.an_lam + 4 * self.an_nu * self.an_nu * self.an_beta)) / (2 * self.an_nu)


        self.info_dict = {'env_name': self.name, 'dim': self.dim, 'nu': self.an_nu, 'lam_congestion': self.lam_congestion}
    
   

    def _sqeuc(self, x):
        return torch.sum(x * x, dim=1, keepdim=True)
    
    def sample_x(self,num_samples):
    
        """
        genearte a  random samples from space.
        
        """
        return -4*torch.rand(size=(num_samples, self.dim))+2  
        
    def sample_xx(self,num_samples):
    
        return torch.linspace(-2,2,num_samples).view(-1,1)
   #------------------------------------------
   # Lagrangian  
   #------------------------------------------ 
   
    def lag(self, tt, xx, qq):
       
        out = 0.5*self._sqeuc( qq) +  0.5*self._sqeuc(xx)     
        return out       
           
   #Hamiltonian
   #------------------------------------------
    def ham(self, tt, xx, pp):
    
        out = 0.5*self._sqeuc( pp) -  0.5*self._sqeuc(xx)     
        return out
  #--------------------------------------------------------
   # calcul Trace    
    def get_trace(self, grad, xx, batch_size, dim, grad_outputs_vec):
    
        hess_stripes = torch.autograd.grad(outputs=grad, inputs=xx,
                                           grad_outputs=grad_outputs_vec,
                                           create_graph=True, retain_graph=True, only_inputs=True)[0]
        pre_laplacian = torch.stack([hess_stripes[i * batch_size: (i + 1) * batch_size, i]
                                     for i in range(0, dim)], dim=1)
        laplacian = torch.sum(pre_laplacian, dim=1)
        laplacian_sum_repeat = laplacian.repeat((1, dim))

        return laplacian_sum_repeat.T
    #--------------------------------------------------------   
    #----------------F-B conditions-------------------------- 
       
    # Initial conition
    def m0(self, x):
        out=((self.an_alpha/(2*math.pi))**(self.dim/2))*torch.exp(-self.an_alpha/(2*self.an_nu)*(torch.sum(x * x, dim=1, keepdim=True)))
        return out
        
    # Terminal condition    
    def g(self, x):
        out=0.5 * self.an_alpha *(torch.sum(x * x, dim=1, keepdim=True))- (self.dim * self.an_alpha * self.an_nu  + (self.an_lam*self.dim)/2 * math.log(self.an_alpha/(2*math.pi*self.an_nu)))
        return out     
     
    #--------------------------------------------------------     
    

   
