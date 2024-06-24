import torch
from utils.utils import one_STRING, two_STRING, cont_STRING
from utils.functions_construct_loss import  get_hjb_loss,get_fp_loss, get_phi_cond_loss,get_rho_cond_loss, get_cont_cond_loss, get_cont_loss
#----------------------------------------------------------------------------------#

# ====================================
#           Construct Loss
# ====================================                                     
                                     

def const_loss(td,tt_samples, rhott_samples, one_or_two_or_cont):

    
    #------------------#
    
    # the HJB equations loss
    hjb_loss_tensor = get_hjb_loss(td, tt_samples, rhott_samples, td['batch_size'],
                                                  td['ones_of_size_phi_out'], td['grad_outputs_vec'])
    # the FP equations loss
    fp_loss_tensor = get_fp_loss(td, tt_samples, rhott_samples, td['batch_size'],
                                          td['ones_of_size_phi_out'], td['grad_outputs_vec'])                                              
                                                  
    # the Control equations_loss                                       
    control_loss_tensor = get_cont_loss(td, tt_samples, rhott_samples, td['batch_size'],
                                                  td['ones_of_size_phi_out'], td['grad_outputs_vec'])
    
                                                  
                                                  
    #--------reduce----------#     
                                            
    hjb_loss_tensor = hjb_loss_tensor[:td['batch_size']]
    fp_loss_tensor = fp_loss_tensor[:td['batch_size']]
    control_loss_tensor = control_loss_tensor[:td['batch_size']] 
    
    
    #loss of HJB
    hjb_loss = ((hjb_loss_tensor)**2).mean(dim=0)
    #loss of fp
    fp_loss = ((fp_loss_tensor)**2).mean(dim=0)
    # loss of control
    cont_loss = ((control_loss_tensor)**2).mean(dim=0)  
    
    
    
    #  computing conditions
    rho_cond = get_rho_cond_loss(td, td['network_two'])
    phi_cond = get_phi_cond_loss(td, td['network_one'])
    cont_cond = get_cont_cond_loss(td, tt_samples, rhott_samples, td['control'], td['network_two'],td['ones_of_size_phi_out'], td['grad_outputs_vec'])
    
    if one_or_two_or_cont == one_STRING:
        
        # Total loss HJB
        total_loss =  hjb_loss + phi_cond  
        
    elif one_or_two_or_cont == two_STRING:
    
        # Total loss FP
        total_loss = fp_loss +  rho_cond 
        
    else:
         # Total loss Control
         total_loss = cont_loss +  cont_cond
        
    if one_or_two_or_cont == one_STRING: 
       
          return total_loss
    
    elif one_or_two_or_cont == two_STRING:
        
          return total_loss
    else:
          return total_loss
     
           
                                     			

