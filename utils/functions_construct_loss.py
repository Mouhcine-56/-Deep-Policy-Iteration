import torch
from utils.utils import uniform_time_sampler, one_STRING, two_STRING
import math

# ================
# functions
# ================

#===========================================HJB====================================================
def get_hjb_loss(td, tt_samples, rhott_samples, batch_size, ones_of_size_phi_out, grad_outputs_vec):

    """
    Get the HJB Loss.
    """
    env = td['env']
    
    rhott_samples = rhott_samples.repeat(repeats=(env.dim, 1))
    tt_samples = tt_samples.repeat(repeats=(env.dim, 1))
    tt_samples.requires_grad_(True)  
    rhott_samples.requires_grad_(True)
    phi_out = td['network_one'](tt_samples, rhott_samples)
    rho_out = td['network_two'](tt_samples, rhott_samples)
    alpha_out = td['control'](tt_samples, rhott_samples)
    phi_grad_tt = torch.autograd.grad(outputs=phi_out, inputs=tt_samples,
                                      grad_outputs=ones_of_size_phi_out,
                                      create_graph=True, retain_graph=True, only_inputs=True)[0]
    phi_grad_xx = torch.autograd.grad(outputs=phi_out, inputs=rhott_samples,
                                      grad_outputs=ones_of_size_phi_out,
                                      create_graph=True, retain_graph=True, only_inputs=True)[0]
    
    phi_trace_xx = env.get_trace(phi_grad_xx, rhott_samples, batch_size, env.dim, grad_outputs_vec)
    
    ham = env.ham(tt_samples, rhott_samples, phi_grad_xx)
    lag=env.lag(tt_samples, rhott_samples, alpha_out)
    
    if env.lam_congestion == 0:
       f0 = 0
    else:
        f0 = env.lam_congestion*torch.log(rho_out)
    
    out = -phi_grad_tt - env.nu * phi_trace_xx + torch.sum(alpha_out * phi_grad_xx , dim=1, keepdim=True)-lag - f0  #  Warning: If env.lam > 0, ensure that NN_rho returns |out| to avoid NaNs during training because of the logarithm.
    
    return out
    
#===================================FP============================================================    
def get_fp_loss(td, tt_samples, rhott_samples, batch_size, ones_of_size_phi_out, grad_outputs_vec):
    """
    Get the FP Loss.
    """
    env = td['env']
    
    rhott_samples = rhott_samples.repeat(repeats=(env.dim, 1))
    tt_samples = tt_samples.repeat(repeats=(env.dim, 1))
    tt_samples.requires_grad_(True)  
    rhott_samples.requires_grad_(True)
    phi_out = td['network_one'](tt_samples, rhott_samples)
    rho_out = td['network_two'](tt_samples, rhott_samples)
    alpha_out = td['control'](tt_samples, rhott_samples)
    phi_grad_tt = torch.autograd.grad(outputs=phi_out, inputs=tt_samples,
                                      grad_outputs=ones_of_size_phi_out,
                                      create_graph=True, retain_graph=True, only_inputs=True)[0]
    phi_grad_xx = torch.autograd.grad(outputs=phi_out, inputs=rhott_samples,
                                      grad_outputs=ones_of_size_phi_out,
                                      create_graph=True, retain_graph=True, only_inputs=True)[0]
    alpha_grad_xx = torch.autograd.grad(outputs=alpha_out, inputs=rhott_samples,
                                      grad_outputs=ones_of_size_phi_out,
                                      create_graph=True, retain_graph=True, only_inputs=True)[0]                                  
    rho_grad_tt = torch.autograd.grad(outputs=rho_out, inputs=tt_samples,
                                      grad_outputs=ones_of_size_phi_out,
                                      create_graph=True, retain_graph=True, only_inputs=True)[0]
    rho_grad_xx = torch.autograd.grad(outputs=rho_out, inputs=rhott_samples,
                                      grad_outputs=ones_of_size_phi_out,
                                      create_graph=True, retain_graph=True, only_inputs=True)[0]  
                                      
    
        
    rho_trace_xx = env.get_trace(rho_grad_xx, rhott_samples, batch_size, env.dim, grad_outputs_vec)
    
    div=torch.sum(rho_grad_xx * alpha_out, dim=1, keepdim=True)   + torch.sum(rho_out * alpha_grad_xx, dim=1, keepdim=True)
    out = (rho_grad_tt - env.nu * rho_trace_xx - div) 

    
    return out

#===================================Control============================================================    
def get_cont_loss(td, tt_samples, rhott_samples, batch_size, ones_of_size_phi_out, grad_outputs_vec):
    """
    Get the Control Loss.
    """
    env = td['env']
    
    rhott_samples = rhott_samples.repeat(repeats=(env.dim, 1))
    tt_samples = tt_samples.repeat(repeats=(env.dim, 1))
    tt_samples.requires_grad_(True) 
    rhott_samples.requires_grad_(True)
    phi_out = td['network_one'](tt_samples, rhott_samples)
    rho_out = td['network_two'](tt_samples, rhott_samples)
    alpha_out = td['control'](tt_samples, rhott_samples)
    phi_grad_tt = torch.autograd.grad(outputs=phi_out, inputs=tt_samples,
                                      grad_outputs=ones_of_size_phi_out,
                                      create_graph=True, retain_graph=True, only_inputs=True)[0]
    phi_grad_xx = torch.autograd.grad(outputs=phi_out, inputs=rhott_samples,
                                      grad_outputs=ones_of_size_phi_out,
                                      create_graph=True, retain_graph=True, only_inputs=True)[0]
    rho_grad_tt = torch.autograd.grad(outputs=rho_out, inputs=tt_samples,
                                      grad_outputs=ones_of_size_phi_out,
                                      create_graph=True, retain_graph=True, only_inputs=True)[0]
    rho_grad_xx = torch.autograd.grad(outputs=rho_out, inputs=rhott_samples,
                                      grad_outputs=ones_of_size_phi_out,
                                      create_graph=True, retain_graph=True, only_inputs=True)[0]  
                                      
    lag=env.lag(tt_samples, rhott_samples, alpha_out)
    
    
    
    out = (lag - torch.sum(phi_grad_xx * alpha_out, dim=1, keepdim=True)) 
    


    return out

   
    


#===================================Conditions============================================================ 

 
def get_rho_cond_loss(td, network_two):

    boud_x = torch.tensor([2], dtype=torch.float).expand((td['batch_size'], 1)).to(td['device'])
    x = (td['env'].sample_x(td['batch_size']).to(td['device'])).requires_grad_(True)
    m_0=td['env'].m0(x).view(-1,1)  
    rho_cond_loss = ((network_two(td['zero'], x)-m_0)**2).mean(dim=0) #+ ((network_two(td['zero'], (-1)*boud_x)-td['env'].m0((-1)*boud_x))**2).mean(dim=0) #+ ((network_two(td['zero'], boud_x)-td['env'].m0(boud_x))**2).mean(dim=0)

    return rho_cond_loss  
    
def get_phi_cond_loss(td, network_one):
    
    
    boud_x = torch.tensor([2], dtype=torch.float).expand((td['batch_size'], 1)).to(td['device'])
    boud_t = torch.tensor([1], dtype=torch.float).expand((td['batch_size'], 1)).to(td['device'])
    x = (td['env'].sample_x(td['batch_size']).to(td['device'])).requires_grad_(True)
    G = td['env'].g(x).view(-1,1)
    phi_cond_loss = ((network_one(td['TT'], x)-G)**2).mean(dim=0) + ((network_one(boud_t, (-1)*boud_x)-td['env'].g((-1)*boud_x))**2).mean(dim=0) + ((network_one(boud_t, boud_x)-td['env'].g(boud_x))**2).mean(dim=0)

    return phi_cond_loss
    
def get_cont_cond_loss(td, tt_samples, rhott_samples, control, network_one, ones_of_size_phi_out, grad_outputs_vec):
    
    if td['env'].nu > 0:  
        rhott_samples = rhott_samples.repeat(repeats=(td['env'].dim, 1))
        tt_samples = tt_samples.repeat(repeats=(td['env'].dim, 1))
    tt_samples.requires_grad_(True)  # WARNING: Keep this after generator evaluation, or else you chain rule generator's time variable
    rhott_samples.requires_grad_(True)
    
    phi_out = td['network_one'](tt_samples, rhott_samples)
    alpha_out = td['control'](tt_samples, rhott_samples)
    
    phi_grad_xx = torch.autograd.grad(outputs=phi_out, inputs=rhott_samples,
                                      grad_outputs=ones_of_size_phi_out,
                                      create_graph=True, retain_graph=True, only_inputs=True)[0]
                                      
    cont_cond_loss = (torch.sum((alpha_out-phi_grad_xx) * (alpha_out-phi_grad_xx), dim=1, keepdim=True)).mean(dim=0)
    
    return cont_cond_loss      
    
    
    
    
    
    
    
    
    
    
    
    
    


