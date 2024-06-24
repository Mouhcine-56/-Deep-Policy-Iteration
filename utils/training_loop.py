from utils.utils import act_funcs, one_STRING, two_STRING, cont_STRING
from Design_model import *
from utils.train_once import train_once
from utils.utils import  Logger
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd 
from scipy.io import savemat
import time
def start_train(a):
    """
    a: A dictionary containing the training arguments
    """
    env = a['env']
    the_logger = Logger(a)
    dim=env.dim
    # =============================================
    #           Precompute some variables
    # =============================================
    # Precompute ones tensor of size phi out for gradient computation
    ones_of_size_phi_out = torch.ones(a['batch_size'] * env.dim, 1).to(a['device']) if env.nu > 0 \
                           else torch.ones(a['batch_size'], 1).to(a['device'])

    # Precompute grad outputs vec for laplacian for Hessian computation
    list_1 = []
    for i in range(env.dim):
        vec = torch.zeros(size=(a['batch_size'], env.dim), dtype=torch.float).to(a['device'])
        vec[:, i] = torch.ones(size=(a['batch_size'],)).to(a['device'])
        list_1.append(vec)
    grad_outputs_vec = torch.cat(list_1, dim=0)

    # ======================================
    #           Setup the learning
    # ======================================
  
    # Make the networks
    network_one = Net_one(dim=env.dim, ns=a['ns'], act_func=act_funcs[a['act_func_1']], hh=a['hh'],
                            device=a['device']).to(a['device'])
                            
    network_two = Net_two(dim=env.dim, ns=a['ns'], act_func=act_funcs[a['act_func_2']], hh=a['hh'], device=a['device'],
                        ).to(a['device'])

    control = Control(dim=env.dim, ns=a['ns'], act_func=act_funcs[a['act_func_3']], hh=a['hh'], device=a['device'],
                        ).to(a['device'])  
                        
    one_optimizer = torch.optim.Adam(network_one.parameters(), lr=a['1_lr'], weight_decay=a['weight_decay'],
                                      betas=a['betas'])
                                      
    two_optimizer = torch.optim.Adam(network_two.parameters(), lr=a['2_lr'], weight_decay=a['weight_decay'],
                                     betas=a['betas'])

    cont_optimizer = torch.optim.Adam(control.parameters(), lr=a['3_lr'], weight_decay=a['weight_decay'],
                                     betas=a['betas'])                                 

    # ===================================
    #           Start iteration
    # ===================================
    # Define initial time and final time constants and some list
    zero = torch.tensor([0], dtype=torch.float).expand((a['batch_size'], 1)).to(a['device'])
    TT = torch.tensor([env.TT], dtype=torch.float).expand((a['batch_size'], 1)).to(a['device'])
    start_time = time.time()
    # Start the iteration
    for epoch in range(a['max_epochs'] + 1):
        # =============================
        #           Info dump
        # =============================
        if epoch % a['print_rate'] == 0:
            print()
            print('-' * 10)
            print(f'--------------\n epoch: {epoch}\n--------------')

            if epoch != 0:
                # Saving neural network and saving to csv
                the_logger.save_nets({'epoch': epoch,
                                      'network_one': network_one,
                                      'network_one_optimizer': one_optimizer,
                                      'network_two': network_two,
                                      'network_two_optimizer': two_optimizer,
                                      'control': control,
                                      'cont_optimizer': cont_optimizer})
                #the_logger.write_training_csv(epoch)

        # ===========================================
        #           Setup training dictionary
        # ===========================================
        train_dict = a.copy()
        train_dict.update({'network_one': network_one,
                           'network_two': network_two,
                           'control': control,
                           'one_optimizer': one_optimizer,
                           'two_optimizer': two_optimizer,
                           'cont_optimizer': cont_optimizer,
                           'ham_func': env.ham,
                           'epoch': epoch,
                           'zero': zero,
                           'TT': TT,
                           'ones_of_size_phi_out': ones_of_size_phi_out,
                           'grad_outputs_vec': grad_outputs_vec,
                           'the_logger': the_logger})
                           
                           
                           
        #initalize parameters for repeat traning 
        i=0
        j=0
        s=0
        
        # ======================================
        #           Train rho/network_2
        # ======================================
        
        #How many network updates 
        while i < 1:
              train_info = train_once(train_dict, two_STRING)
        
              the_logger.log_training(train_info, two_STRING)
              
              i=i+1
         
            
        if epoch % a['print_rate'] == 0:
            the_logger.print_to_console(train_info, two_STRING)
            
            # Save Loss  of  network_2
            #loss_net2.append(train_info['total_loss'])
            
        # ===========================================
        #           Train phi/network_1
        # ===========================================
        
        #How many network updates 
        while j < 1:
	
              train_info = train_once(train_dict, one_STRING)

              the_logger.log_training(train_info, one_STRING)
              
              j=j+1
               
        if epoch % a['print_rate'] == 0:
            the_logger.print_to_console(train_info, one_STRING)
            
            # Save Loss of network_1
            #loss_net1.append(train_info['total_loss'])  
            
	
	# ======================================
        #           Train alpha/control
        # ======================================
        
        #How many network updates 
        while s < 1:
        
              train_info = train_once(train_dict, cont_STRING)
        
              the_logger.log_training(train_info, cont_STRING)
              
              s=s+1
            
        if epoch % a['print_rate'] == 0:
            the_logger.print_to_console(train_info, cont_STRING)
        
                   
        # =======================================
        #     Relative_Error and  Plot solution.
        # =======================================
        
        if epoch % a['print_rate'] == 0:
            elapsed = time.time() - start_time
            print('Time:', elapsed)
            #Time.append(elapsed)
            
            #plot solution 
            #the_logger.show1(network_one,network_two)  
            #the_logger.show2(network_one,network_two)  
            #the_logger.show3(network_one,network_two)
            #plt.close() 
            
            #Compute rel_error
            Err_phi,Err_rho,ep=the_logger.show_rel_Err(train_dict, network_one,network_two,epoch)
            

    
    return the_logger
