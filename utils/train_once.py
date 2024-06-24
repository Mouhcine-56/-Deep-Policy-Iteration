import torch
from utils.construct_loss import const_loss
from utils.utils import one_STRING, two_STRING,cont_STRING,set_requires_grad, set_zero_grad, get_samples, optimizer_step, Logger

# ====================================
#           The main trainer
# ====================================

def train_once(td, one_or_two_or_cont):

    """
    Trains the Net1, Net2  and Control.
    """
    error_msg = f'Invalid one_or_two_or_cont. Should be {one_STRING} or {two_STRING} or {cont_STRING} but got: {one_or_two_or_cont}'
    assert (one_or_two_or_cont == one_STRING or one_or_two_or_cont == two_STRING, one_or_two_or_cont == cont_STRING), error_msg
    

    # Activate computing computational graph of Net1/Net2/Control
    set_requires_grad(td['network_one'], td['network_two'], td['control'],  one_or_two_or_cont)

    # Zero the gradients
    set_zero_grad(td['one_optimizer'], td['two_optimizer'], td['cont_optimizer'], one_or_two_or_cont)

    # Get samples
    tt_samples, rhott_samples = get_samples(td, one_or_two_or_cont)
    
    #Construct loss
    if one_or_two_or_cont == one_STRING:
          total_loss=const_loss(td, tt_samples, rhott_samples, one_or_two_or_cont)
          
    elif one_or_two_or_cont == two_STRING:
          total_loss=const_loss(td, tt_samples, rhott_samples, one_or_two_or_cont)
    else: 
          total_loss=const_loss(td, tt_samples, rhott_samples, one_or_two_or_cont) 
             
    # Backprop and optimize
    total_loss.backward()
    
    #Update parameter 
    optimizer_step(td['one_optimizer'], td['two_optimizer'], td['cont_optimizer'], one_or_two_or_cont)

    # Get info about the training
    with torch.no_grad():
        
        training_info = {'total_loss': total_loss.item()}
        
    return training_info
