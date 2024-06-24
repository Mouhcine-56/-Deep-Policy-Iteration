import torch
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import datetime
import os
from os.path import join
import pprint as pp
import pickle
import csv
import imageio


# =======================================================
#           Utility functions and miscellaneous
# =======================================================
one_STRING = 'net1'
two_STRING = 'net2'
cont_STRING = 'net3'

# Some activation functions
act_funcs = {'tanh': lambda x: torch.tanh(x),
             'relu': lambda x: torch.relu(x),
             'leaky_relu': lambda x: torch.nn.functional.leaky_relu(x),
             'softplus': lambda x: torch.nn.functional.softplus(x)}


def sqeuc(xx):
    return torch.sum(xx * xx, dim=1, keepdim=True)


def uniform_time_sampler(batch_size):
    return torch.rand(size=(batch_size, 1))
    
def set_requires_grad(network_one, network_two, control, one_or_two_or_cont):

    """
    Turn on requires_grad for the both Networks.
    """
    if one_or_two_or_cont == one_STRING:
        for param in network_one.parameters():
            param.requires_grad_(True)
        for param in network_two.parameters():
            param.requires_grad_(False)
        for param in control.parameters():
            param.requires_grad_(False)        
    elif one_or_two_or_cont == two_STRING:
        for param in network_one.parameters():
            param.requires_grad_(False)
        for param in network_two.parameters():
            param.requires_grad_(True)
        for param in control.parameters():
            param.requires_grad_(False)        
    elif one_or_two_or_cont == cont_STRING:
        for param in network_one.parameters():
            param.requires_grad_(False)
        for param in network_two.parameters():
            param.requires_grad_(False) 
        for param in control.parameters():
            param.requires_grad_(True)         
            
    else:
        raise ValueError(f'Invalid one_or_two. Should be {one_STRING} or {two_STRING} but got: {one_or_two}')


def set_zero_grad(one_optimizer, two_optimizer, cont_optimizer, one_or_two_or_cont):

    """
    Zero the gradients for the one we're training.
    """
    if one_or_two_or_cont == one_STRING:
        one_optimizer.zero_grad()
    elif one_or_two_or_cont == two_STRING:
        two_optimizer.zero_grad()
    elif one_or_two_or_cont == cont_STRING:    
        cont_optimizer.zero_grad()
    else:
        raise ValueError(f'Invalid one_or_two. Should be {one_STRING} or {two_STRING} but got: {one_or_two}')


def do_grad_clip(network_one, network_two, clip_value, one_or_two):
    """
    Clips the gradient of the discriminator and/or generator.
    """
    if disc_or_gen == DISC_STRING:
        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=clip_value)
        torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=clip_value)
    elif disc_or_gen == GEN_STRING:
        torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=clip_value)
        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=clip_value)
    else:
        raise ValueError(f'Invalid disc_or_gen. Should be {DISC_STRING} or {GEN_STRING} but got: {disc_or_gen}')


def optimizer_step(one_optimizer, two_optimizer, cont_optimizer, one_or_two_or_cont):

    """
    Take a step of the network_one or network_two optimizers.
    """
    if one_or_two_or_cont == one_STRING:
        one_optimizer.step()   
    elif one_or_two_or_cont == two_STRING:
        two_optimizer.step()
    elif one_or_two_or_cont == cont_STRING:
        cont_optimizer.step()
    else:
        raise ValueError(f'Invalid one_or_two. Should be {one_STRING} or {two_STRING} but got: {one_or_two}')

# =====================
# Get samples
# =====================

def get_samples(td, one_or_two_or_cont):
   
    #rho00 = td['env'].sample_rho0(td['batch_size']).to(td['device'])
    tt_samples = (torch.linspace(0,1,td['batch_size'])).to(td['device'])

    if one_or_two_or_cont == one_STRING:
        rhott_samples = (td['env'].sample_x(td['batch_size']).to(td['device'])).requires_grad_(True)
    elif one_or_two_or_cont == two_STRING:
        rhott_samples = (td['env'].sample_x(td['batch_size']).to(td['device'])).requires_grad_(True)
    elif one_or_two_or_cont == cont_STRING:  
        rhott_samples = (td['env'].sample_x(td['batch_size']).to(td['device'])).requires_grad_(True)  
    else:
        raise ValueError(f'Invalid one_or_two. Should be \'one\' or \'two\' but got: {one_or_two}')

    return tt_samples.view(-1,1), rhott_samples       

    

# ===========================================================
#           Helper function 
# ===========================================================
def _get_time_string():
    """
    Get the current time in a string
    """
    out = str(datetime.datetime.now()).replace(':', '-').replace(' ', '-')[:-7]
    out = out[:10] + '__' + out[11:]  # separate year-month-day from hour-minute-seconds

    return out

# ================================
#           Logger class
# ================================
class Logger(object):

    """
    A logger to log training.
    """
    def __init__(self, args):
        self.do_logging = args['do_logging']
        self.device = args['device']
        if self.do_logging:
            # Make the experiment directory
            run_dir = 'Run_' + _get_time_string() + '_' + args['experiment_name']
            self.experiment_dir = join('Experiments', run_dir)

            

            # Make CSV file for logging
            self.one_csv_filepath = join(self.experiment_dir, 'one_train_log.csv')
            self.two_csv_filepath = join(self.experiment_dir, 'two_train_log.csv')
            self.dict_of_csv_filepaths = {one_STRING: self.one_csv_filepath,
                                          two_STRING: self.two_csv_filepath}

      
            self.train_log_dict = {one_STRING: {}, two_STRING: {},  cont_STRING: {}}

            # Print rate
            self.print_rate = args['print_rate']

            # Print/save training hyperparameters
            self.args_dir = join(self.experiment_dir, 'args')
            os.makedirs(self.args_dir)
            with open(join(self.args_dir, 'experiment_args.txt'), 'w') as ff:
                pp.pprint(args, ff)
            with open(join(self.args_dir, 'experiment_args.pkl'), 'wb') as ff:
                pickle.dump(args, ff)
            # Save environment parameters
            self.env = args['env']
            with open(join(self.args_dir, 'env_args.txt'), 'w') as ff:
                pp.pprint(self.env.info_dict, ff)
            with open(join(self.args_dir, 'env_args.pkl'), 'wb') as ff:
                pickle.dump(self.env.info_dict, ff)
                
            
    # plot 
    #--------------------------------        
    def show1(self, network_one,network_two):
        with torch.no_grad():
             x=torch.linspace(-2,2,100).view(-1,1).to(self.device)
             the_timepoint = torch.tensor([0],dtype=torch.float)
             t=the_timepoint[0].expand(x.shape[0], 1).to(self.device)
             network_one.eval()
             network_two.eval()   
             y1=network_one(t,x).cpu().detach().numpy() # phi: Aproximation Function at t=0
             y2=network_two(t,x).cpu().detach().numpy() # rho: Aproximation Function at t=0
             x=x.cpu().detach().numpy()    
             plt.plot(x,y1)
             plt.plot(x,y2)
             plt.savefig('pho1.png')
     #--------------------------------        
    def show2(self, network_one,network_two):
        with torch.no_grad():
             x=torch.linspace(-2,2,100).view(-1,1).to(self.device)
             the_timepoint = torch.tensor([0.5],dtype=torch.float)
             t=the_timepoint[0].expand(x.shape[0], 1).to(self.device)
             network_one.eval()
             network_two.eval()   
             y1=network_one(t,x).cpu().detach().numpy() 
             y2=network_two(t,x).cpu().detach().numpy() 
             x=x.cpu().detach().numpy()       
             plt.plot(x,y1)
             plt.plot(x,y2) 
             plt.savefig('pho2.png')
    #--------------------------------
    def show3(self, network_one,network_two):
        with torch.no_grad():
             x=torch.linspace(-2,2,100).view(-1,1).to(self.device)
             the_timepoint = torch.tensor([1],dtype=torch.float)
             t=the_timepoint[0].expand(x.shape[0], 1).to(self.device)
             network_one.eval()
             network_two.eval()   
             y1=network_one(t,x).cpu().detach().numpy() 
             y2=network_two(t,x).cpu().detach().numpy() 
             x=x.cpu().detach().numpy()       
             plt.plot(x,y1)
             plt.plot(x,y2)
             plt.savefig('pho3.png')                  
    #--------------------------------
    #comput Error Relative     
    def show_rel_Err(self, td, network_one, network_two,epoch):
        
        env = td['env']
        def sqeuc( x):
            return torch.sum(x * x, dim=1, keepdim=True)
            
        #Exact solution     
        def phi(x,t):
            out=0.5 * env.an_alpha *(torch.sum(x * x, dim=1, keepdim=True))- (env.dim * env.an_alpha * env.an_nu  + (env.an_lam*env.dim)/2 * math.log(env.an_alpha/(2*math.pi*env.an_nu)))*t
            return out
        def rho(t,x):
            out=((env.an_alpha/(2*math.pi))**(env.dim/2))*torch.exp(-env.an_alpha/(2*env.an_nu)*(torch.sum(x * x, dim=1, keepdim=True)))
            return out   
            
                    
        with torch.no_grad():
            x=np.linspace(-2, 2,100)
            t=np.linspace(0, 1,100)
            X, T = np.meshgrid(x,t)
            X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
            x1 = torch.tensor(X_star[:, 0:1]).float().to(self.device)
            t1 = torch.tensor(X_star[:, 1:2]).float().to(self.device)
            network_one.eval()
            network_two.eval()
            
            # compute Err
            y1=phi(x1,t1).cpu().detach().numpy()
            
            y2=network_one(t1,x1).cpu().detach().numpy()
        
            E1=np.linalg.norm(y1-y2,2)/np.linalg.norm(y1,2)
            rho1=rho(t1,x1).cpu().detach().numpy()
            
            rho2=network_two(t1,x1).cpu().detach().numpy()
            E2=np.linalg.norm(rho1-rho2,2)/np.linalg.norm(rho1,2)
            
            print('Error_relative phi:',E1)
            print('Error_relative rho:',E2) 
            return E1,E2,epoch 
            
    #=========================================================             
    #compute mean of distribution
    def mean(self,X): #X matrix
        m,n=X.size() #m=row ,   n=colons
        x=[]
        y=torch.zeros(n)
        for i in range(m):
            x.append(X[i])
            y=y+x[i]
        mu=1/m*y
        mu=mu.view(1, -1)
        return mu
    #--------------------------------
    #compute variance of distribution
    def variance(self,mu,X):
        m,n=X.size() #m=row ,   n=colons
        x_mu=[]
        Y=torch.zeros(n,n)
        for i in range(m):
            x_mu.append(X[i].view(1, -1)-mu)
            Y=Y+torch.transpose(x_mu[i], 0, 1)*x_mu[i]
        var=1/m*Y  
        return var
     #--------------------------------   
    #compute the relative entrepy
    def KL_div(self,X1,X2,dim):
        mu0=self.mean(X1)
        mu1=self.mean(X2)
        var0=self.variance(mu0,X1)
        var1=self.variance(mu1,X2)
        mu0=mu0.view(-1, 1)
        mu1=mu1.view(-1, 1)
        var1_inv=torch.inverse(var1)
        x0=mu1-mu0
        x1=torch.transpose(x0,0,1)
        x2=torch.matmul(x1,var1_inv)
        kl1=torch.trace(torch.matmul(var1_inv, var0))
        kl2=torch.matmul(x2,x0)-dim
        y0=torch.det(var1)
        y1=torch.det(var0)
        if y1==0:
           kl3=0
        else:
             kl3=math.log(y0/y1)
        KL=1/2*(kl1+kl2+kl3)
        return KL 
     #--------------------------------     
    # Kernel density estimation
    def Kernel_DE(self,vect,h,dim):
            dist = torch.einsum('ijk, ijk->ij', vect, vect)  # pairwise_distance
            const_part = math.pow(1 / h, dim) * math.pow(1 / (2 * math.pi * 0.01), dim / 2)
            exp_part = torch.exp(-(1 / (2 * 0.01)) * dist)
            dist_exp = const_part * exp_part
            # pw_dist_exp_log = torch.log(pw_dist_exp)
            dist_exp_mean = dist_exp.mean(dim=0).unsqueeze(1)
            # if any(pw_dist_exp_mean > 0):
            #     print('pw_dist_exp_mean:', pw_dist_exp_mean)
            return dist_exp_mean
             
    #=========================================================  
             
    def _initialize_dict_of_lists(self, source_dict, one_or_two_or_cont):
        """
        Create a new dictionary with the same keys as the source dictionary,
        and turn the values into lists.
        """
        self.train_log_dict[one_or_two_or_cont] = source_dict.copy()
        for key, val in self.train_log_dict[one_or_two_or_cont].items():
            self.train_log_dict[one_or_two_or_cont][key] = [val]

    def log_training(self, training_info_dict, one_or_two_or_cont):
        """
        Append stuff to the list for logging
        the_dict: Information about the training.
        """
        if self.do_logging:
            # If the dictionary of lists is empty, then initialize it
            if len(self.train_log_dict[one_or_two_or_cont]) == 0:
                self._initialize_dict_of_lists(training_info_dict, one_or_two_or_cont)
            else:
                for key, val in training_info_dict.items():
                    self.train_log_dict[one_or_two_or_cont][key].append(val)

    def _initialize_training_csv(self, csv_path, one_or_two_or_cont):
        """
        Essentially creates the csv, and writes the header.
        """
        key_list = ['epoch'] + list(self.train_log_dict[one_or_two_or_cont].keys())
        with open(csv_path, 'w') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(key_list)

    def write_training_csv(self, epoch):
        """
        Write to the csv file.
        """
        if self.do_logging:
            for STRING in [one_STRING, two_STRING, cont_STRING]:
                csvpath = self.dict_of_csv_filepaths[STRING]
                # Initialize the csv file
                if not os.path.exists(csvpath):
                    self._initialize_training_csv(csvpath, STRING)
                # Create a dictionary to print into a csv file
                dict_of_avg = {'epoch': epoch}
                for key, val in self.train_log_dict[STRING].items():
                    dict_of_avg[key] = np.mean(val[-self.print_rate:]) if val[0] != '--' else '--'
                # Write to the network_one or network_two training log file
                with open(csvpath, 'a') as csvfile:
                    csv_writer = csv.writer(csvfile)
                    csv_writer.writerow(list(dict_of_avg.values()))

    def save_nets(self, the_dict):
        """
        Save the network_one and generator models,
        and their optimizers.
        """
        if self.do_logging:
            self.model_path = join(self.experiment_dir, 'models')
            os.makedirs(self.model_path, exist_ok=True)

            epoch = the_dict['epoch']
            network_one = the_dict['network_one']
            network_one_optimizer = the_dict['network_one_optimizer']
            network_two = the_dict['network_two']
            network_two_optimizer = the_dict['network_two_optimizer']

            # Save network_one
            torch.save({'epoch': epoch,
                        'model_state_dict': network_one.state_dict(),
                '       optimizer': network_one_optimizer.state_dict()},
                join(self.model_path, f'network_one-epoch-{epoch}.pth.tar'))

            # Save network_two
            torch.save({'epoch': epoch,
                        'model_state_dict': network_two.state_dict(),
                        'optimizer': network_two_optimizer.state_dict()
                        },
                join(self.model_path, f'network_two-epoch-{epoch}.pth.tar'))



    def print_to_console(self, td, one_or_two_or_cont):
        """
        Stuff to print to the console.
        """
        with torch.no_grad():
        
            # Setup variables
            error_msg = f'Invalid one_or_two_or_cont. Should be {one_STRING} or {two_STRING} or {cont_STRING} but got: {one_or_two_or_cont}'
      
            # Start printing to the console
            if one_or_two_or_cont == one_STRING:
                print('#==================#\nNETWORK_one losses: \n#==================#')
            elif one_or_two_or_cont == two_STRING:
                print('#==================#\n NETWORK_two losses:\n#==================#')
            elif one_or_two_or_cont == cont_STRING:   
                print('#==================#\n Control losses:\n#==================#')
            else:
                raise ValueError(error_msg)
            
            print('total_loss:', td['total_loss'])

            print()
