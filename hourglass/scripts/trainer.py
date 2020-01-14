from torchvision import transforms
from comet_ml import Experiment
import torch
import torch.nn as nn
import os
import torch.nn.functional as F
from PIL import Image
import network


class Height_trainer(nn.Module):
    def __init__(self, hyperparameters):
        super(Height_trainer, self).__init__()
        
        lr = hyperparameters["lr"]
        comet_exp = hyperparameters["comet_exp"]
        comet_key = hyperparameters['comet_key']
        self.transforms = hyperparameters["transforms_params"]
        self.batch_size = hyperparameters['batch_size']
        self.max_epochs = hyperparameters['epochs']
        self.leaky_lambda = hyperparameters['leaky_lambda']
        self.leaky_negative_slope = hyperparameters['leaky_negative_slope']
        self.mask_val = hyperparameters['mask_val']
            
        self.model = network.Model().cuda()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        
            
        if comet_exp:
            self.exp = Experiment(comet_key, project_name=hyperparameters["project_name"], auto_metric_logging=False)
            
            
        
    def L1_loss(self, input, target):
        """
        Compute pixelwise MAE between two images input and target

        Arguments:
            input {torch.Tensor} -- Image tensor
            target {torch.Tensor} -- Image tensor

        Returns:
            torch.Float -- pixelwise L1 loss
        """
        return torch.mean(torch.abs(input - target))
    
    def L1_loss_mask(self, input, target, mask):
        """
        Compute a weaker version of the MAE between two tensors input and target 
        where the L1 is only computed on the unmasked region

        Arguments:
            input {torch.Tensor} -- tensor
            target {torch.Tensor} -- tensor 
            mask {torch.Tensor} -- binary Mask of size HxW 

        Returns:
            torch.Float -- L1 loss over input.(1-mask) and target.(1-mask)
        """
        return torch.mean(torch.abs(torch.mul((input - target), 1 - mask)))
    
    def leaky_relu_loss(self, input, target, mask):
        """
        Penalize masked area less than the rest

        """
        dif = input - target
        leak = F.leaky_relu(torch.mul(dif, mask), self.leaky_negative_slope)
        loss = torch.mul(dif,(1-mask))**2 + self.leaky_lambda * leak
        return(torch.mean(loss))
    
    def L2_loss(self, input, target):
        """
        Compute pixelwise MAE between two images input and target

        Arguments:
            input {torch.Tensor} -- Image tensor
            target {torch.Tensor} -- Image tensor

        Returns:
            torch.Float -- pixelwise L2 loss
        """
        return torch.mean((input - target)**2)
    
    def L2_loss_mask(self, input, target, mask):
        """
        Compute a weaker version of the MSE between two tensors input and target 
        where the L2 is only computed on the unmasked region

        Arguments:
            input {torch.Tensor} -- tensor
            target {torch.Tensor} -- tensor 
            mask {torch.Tensor} -- binary Mask of size HxW 

        Returns:
            torch.Float -- L2 loss over input.(1-mask) and target.(1-mask)
        """
        return torch.mean(torch.abs(torch.mul((input - target)**2, 1 - mask)))
    
    def get_mask_from_val(self, input, val):
        return (input == val) 
    
    def get_mask_above_threshold(self, input, val):
        return (input > val)
   
       
    def model_update(
        self, inputs, hyperparameters, gt, mask, comet_exp=None, mask_sky = None
    ):
        """
        Update the generator parameters
        Arguments:
            inputs {torch.Tensor} -- Input images tensor format
            hyperparameters {dictionnary} -- dictionnary with all hyperparameters 
        Keyword Arguments:
            mask_a {torch.Tensor} -- binary mask (0,1) corresponding to the ground in x_a (default: {None})
            mask_b {torch.Tensor} -- binary mask (0,1) corresponding to the water in x_b (default: {None})
            comet_exp {cometExperience} -- CometML object use to log all the loss and images (default: {None})
            synth {boolean}  -- binary True or False stating if we have a synthetic pair or not 
        Returns:
            [type] -- [description]
        """
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        
        self.loss = self.leaky_relu_loss(outputs, gt, mask)       
        
        self.loss.backward()
        self.optimizer.step()
        
        if comet_exp is not None:
            self.exp.log_metric("leaky_loss", self.loss.cpu().detach())
            if not mask_sky is None:
                self.exp.log_metric("L2_loss_mask", self.L2_loss_mask(outputs, gt, mask_sky).cpu().detach())
                self.exp.log_metric("L1_loss_mask", self.L1_loss_mask(outputs, gt, mask_sky).cpu().detach())
            else:
                self.exp.log_metric("L2_loss", self.L2_loss(outputs, gt).cpu().detach())
                self.exp.log_metric("L1_loss", self.L1_loss(outputs, gt).cpu().detach())


    def sample(self, batch_input):
        """ 
        Infer the model on a batch of image
        
        Arguments:
            batch_input {torch.Tensor} -- batch of input images as tensor
        
        Returns:
            torch tensor of heights -- 
        """
        self.eval()
        output = self.model(batch_input)
        self.train()
        return(output) 
        
"""    
    def resume(self, checkpoint_dir, hyperparameters):
        
        Resume the training loading the network parameters
        
        Arguments:
            checkpoint_dir {string} -- path to the directory where the checkpoints are saved
            hyperparameters {dictionnary} -- dictionnary with all hyperparameters 
        
        Returns:
            int -- number of iterations (used by the optimizer)
        
        # Load model
        last_model_name = get_model_list(checkpoint_dir, "height")
        state_dict = torch.load(last_model_name)
        
        iterations = int(last_model_name[-11:-3])
        # Load discriminators
        last_model_name = get_model_list(checkpoint_dir, "dis")
        state_dict = torch.load(last_model_name)
        self.dis_a.load_state_dict(state_dict["a"])
        self.dis_b.load_state_dict(state_dict["b"])
        # Load optimizers
        state_dict = torch.load(os.path.join(checkpoint_dir, "optimizer.pt"))
        self.dis_opt.load_state_dict(state_dict["dis"])
        self.gen_opt.load_state_dict(state_dict["gen"])

        if self.domain_classif_ab == 1:
            self.dann_opt.load_state_dict(state_dict["dann"])
            self.dann_scheduler = get_scheduler(
                self.dann_opt, hyperparameters, iterations
            )
        # Reinitilize schedulers
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters, iterations)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters, iterations)
        print("Resume from iteration %d" % iterations)
        return iterations
        

"""    