"""
Implementation of diffusion model in pytorch to generate MNIST images.

Based on https://huggingface.co/blog/annotated-diffusion,
which implements the DDPM paper https://arxiv.org/abs/2006.11239.
"""

import os
import sys
import pathlib
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm.auto import tqdm
import pickle

import torch
import torchvision

from architectures import unet
import plot_results

import common_base

# ================================================================================================
# Try to implement set-to-set GAN
# ================================================================================================
class DDPM_JetImage(common_base.CommonBase):
    def __init__(self, results, model_params, jetR, device, output_dir):

        self.model_params = model_params
        self.jetR = jetR
        self.device = device
        self.output_dir = output_dir
        
        self.initialize_data(results)

        self.results_folder = pathlib.Path(f"{self.output_dir}/results")
        self.results_folder.mkdir(exist_ok = True)

        self.plot_folder = pathlib.Path(f"{self.output_dir}/plot")
        self.plot_folder.mkdir(exist_ok = True)

        print(self)

    # -----------------------------------------------------------------------
    # Initialize data to the appropriate format
    # -----------------------------------------------------------------------
    def initialize_data(self, results):

        print('------------------ Dataset ------------------')
        # Construct Dataset class
        self.image_dim = self.model_params['image_dim']
        self.n_train = self.model_params['n_train']
        train_dataset = JetImageDataset(results[f'jet__{self.jetR}__hadron__jet_image__{self.image_dim}'],results[f'jet__{self.jetR}__parton__jet_image__{self.image_dim}'],
                                        self.n_train)
        #Add here partons and filter before
        # Construct a dataloader
        self.batch_size = self.model_params['batch_size']
        self.train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)  
        print(f'  Total samples: {len(train_dataset)} (train)')
        print('--------------------------------------------')

    # -----------------------------------------------------------------------
    # Train model
    # -----------------------------------------------------------------------
    def train(self):

        #---------------------------------------------
        # Define the diffusion pipeline.
        #
        # Starting with a number of time steps [0,...,T] and a noise schedule beta_0<...<beta_T:
        #
        # We first compute the forward diffusion process using a reparameterization of the beta's:
        #   alpha_t=1-beta_t, \bar{alpha}=\prod_s=1^t{alpha_s}.
        # This allows us to directly diffuse to a time step t:
        #   q(x_t|x_0) = N(sqrt(\bar{alpha}_t) x_0, 1-\bar{alpha}_t I), which is equivalent to
        #   x_t = sqrt(1-beta_t) x_{t-1} + sqrt(beta_t) epsilon, where epsilon ~N(0,I).
        #
        # We then want to compute the reverse diffusion process, p(x_{t-1} | x_t).
        # This is analytically intractable (the image space is too large), so we use a learned model to approximate it.
        # We assume that p is Gaussian, and assume a fixed variance sigma_t^2=beta_t for each t. 
        # We then reparameterize the mean and can instead learn a noise eps_theta:
        #   Loss = MSE[eps, eps_theta(x_t,t)], where eps is the generated noise from the forward diffusion process
        #                                      and eps_theta is the learned function.      
        # 
        # That is, the algorithm is:
        #  1. Take a noiseless image x_0
        #  2. Sample a time step t
        #  3. Sample a noise term epsilon~N(0,I), and forward diffuse
        #  4. Perform gradient descent to learn the noise: optimize MSE[eps, eps_theta(x_t,t)]
        #
        # The NN takes a noisy image as input, and outputs an image of the noise.
        # 
        # We can then construct noise samples, and use the NN to denoise them and produce target images. 
        #---------------------------------------------

        # Define beta schedule, and define related parameters: alpha, alpha_bar
        self.T = 300
        
        self.beta = torch.linspace(0.0001, 0.02, self.T)
        alpha = 1. - self.beta
        alphabar = torch.cumprod(alpha, axis=0)
        alphabar_prev = torch.nn.functional.pad(alphabar[:-1], (1, 0), value=1.0)
        self.sqrt_1_alpha = torch.sqrt(1.0 / alpha)
       

        # Quantities needed for diffusion q(x_t | x_{t-1})
        self.sqrt_alphabar = torch.sqrt(alphabar)
        self.sqrt_one_minus_alphabar = torch.sqrt(1. - alphabar)
        self.sqrt_1_minalphabar = torch.sqrt(1.0/(1-alphabar))
        # Quantities needed for inversion q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.beta * (1. - alphabar_prev) / (1. - alphabar)
        self.st_coef = -torch.sqrt(alpha)*(1. - alphabar_prev) / (1. - alphabar)
        #Enet
        class ENet(torch.nn.Module):
            def __init__(self, dim, hidden_dim=50):
                super().__init__()
                self.dnn_stack = torch.nn.Sequential(
                    torch.nn.Linear(dim, 50),
                    torch.nn.ReLU(),
                    torch.nn.Linear(50, 784)
                )

            def forward(self, x):
                return self.dnn_stack(x)
        E = ENet(784)
        E.to(self.device)
        # Forward diffusion example
        #imgs = next(iter(self.train_dataloader))
        #img0 = imgs[0][0]
        #print(img0.shape)
        #img0 = img0.to(self.device)
        #difuze = self.q(E,img0, torch.tensor([self.T-1]),1,torch.rand(img0.shape),torch.rand(img0.shape))
        #print('difuze size',difuze.shape)
        #plt.imshow(img0[0].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
        #plt.savefig(f"{self.plot_folder}/img0.png")
        #plt.imshow(difuze[0].detach().cpu().numpy(), cmap='gray', vmin=0, vmax=1)
        #plt.savefig(f"{self.plot_folder}/img0_diffused.png")
        #plt.clf()
        #print('I saved example')

        #---------------------------------------------
        # Training the denoising model
        #---------------------------------------------
        print()
        print('------------------- Model -------------------')
        # Expects 4D tensor input: (batch, channels, height, width)
        model = unet.Unet(
            dim=self.image_dim,
            channels=1,
            dim_mults=(1, 2, 4,)
        )
        model.to(self.device)
        self.count_parameters(model)

        print()
        print('------------------ Training ------------------')
        # Hyperparameters
        learning_rate = self.model_params['learning_rate']
        n_epochs = self.model_params['n_epochs']
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        opt = torch.optim.Adam(E.parameters(), lr=learning_rate)

        model_outputfile = str(self.results_folder / 'model.pkl')
        if os.path.exists(model_outputfile):
            model.load_state_dict(torch.load(model_outputfile))
            print(f"Loaded trained model from: {model_outputfile} (delete and re-run if you'd like to re-train)")
        else:
            training_loss = []
            for epoch in range(n_epochs):
                print(f'Epoch {epoch}')
                for step, (X,C) in enumerate(self.train_dataloader):

                    X = X.to(self.device)
                    C = C.to(self.device)
                    size = C.shape[0]
                    #print(C.shape,'conditions')
                    #print(X.shape,'hadrons')
    
                    # Algorithm 1 line 3: sample t uniformally for every example in the batch
                    t = torch.randint(0, self.T, size=(1,), device=self.device).long()
                    loss = self.p_losses( model,E, X, t,size,C, noise=None)
                    training_loss.append(loss.cpu().detach().numpy().item())
                    if step % 100 == 0:
                        print(f"  Loss (step {step}):", loss.item())

                    loss.backward()
                    optimizer.step()
                    opt.step
                    optimizer.zero_grad()
                    opt.zero_grad()
            print('Done training!')
            torch.save(model.state_dict(), model_outputfile)
            print(f'Saved model: {model_outputfile}')
            plt.plot(training_loss)
            plt.xlabel('Training step')
            plt.ylabel('Loss')
            plt.savefig(str(self.plot_folder / f'loss.png'))
            print('I saved loss plot')
            plt.clf()
   
        #---------------------------------------------
        # Sampling from the trained model
        #---------------------------------------------
        print()
        print('------------------ Sampling ------------------')
        print('--------------------------------------------')
        n_samples = 1000
        test_data = self.train_dataloader.dataset
        C = test_data.label
        c = C[:n_samples].view(n_samples,784)
        print(c.shape,'conditions shape')
        samples_outputfile = str(self.results_folder / 'samples.pkl')
        if os.path.exists(samples_outputfile):
            with open(samples_outputfile, "rb") as f:
                samples = pickle.load(f) 
            print(f"Loaded samples from: {samples_outputfile} (delete and re-run if you'd like to re-train)")
        else:
            samples = self.sample(model,E,c, image_size=self.image_dim, n_samples=n_samples)

            with open(samples_outputfile, "wb") as f:
                pickle.dump(samples, f)
            print(f'Saved {n_samples} samples: {samples_outputfile}')

        # Get the generated images (i.e. last time step)
        samples_0 = np.squeeze(samples[-1])
 
        #---------------------------------------------
        # Plot some observables
        #---------------------------------------------  
        print()
        print('------------------ Plotting ------------------')
        print('--------------------------------------------')

        # Get images from the training set, for comparison
        test_had = test_data.data 
        indices = torch.randperm(len(test_had))[:n_samples]
        train_had = np.squeeze(torch.stack([test_had[idx] for idx in indices]).numpy())

        # Now, samples_0 and train_samples are of shape (n_samples, image_dim, image_dim)
   
        # N pixels above threshold
        threshold = 1.e-2
        N_generated = np.sum(samples_0 > threshold, axis=(1,2))
        N_train = np.sum(train_had > threshold, axis=(1,2))
        plot_results.plot_histogram_1d(x_list=[N_generated, N_train], 
                                       label_list=['generated', 'target'],
                                       bins=np.linspace(-0.5, 29.5, 31),
                                       xlabel=f'N pixels with z>{threshold}', 
                                       filename='N_pixels.png', 
                                       output_dir=self.plot_folder)

        # z distribution
        z_generated = samples_0.flatten()
        z_train = train_had.flatten()
        plot_results.plot_histogram_1d(x_list=[z_generated, z_train], 
                                       label_list=['generated', 'target'],
                                       bins=np.linspace(0., 1., 101),
                                       logy=True,
                                       xlabel=f'z (pixels)', 
                                       filename='z.png', 
                                       output_dir=self.plot_folder)

        # Plot some sample images
        for random_index in range(3):

            plt.imshow(samples_0[random_index].reshape(self.image_dim, self.image_dim, 1), cmap="gray")
            plt.savefig(str(self.plot_folder / f'{random_index}_generated.png'))
            plt.clf()

            # Generate a gif of denoising
            fig = plt.figure()
            ims = []
            for i in range(self.T):
                im = plt.imshow(samples[i][random_index].reshape(self.image_dim, self.image_dim, 1), cmap="gray", animated=True)
                ims.append([im])
            animate = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
            animate.save(str(self.plot_folder / f'{random_index}_generated.gif'))

      

    # -----------------------------------------------------------------------
    #
    # -----------------------------------------------------------------------
    def count_parameters(self, model):
        total_params = sum(p.numel() for p in model.parameters())
        total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params}")
        print(f"Trainable parameters: {total_trainable_params}")

    # -----------------------------------------------------------------------
    # Defining the loss function
    # -----------------------------------------------------------------------
    def p_losses(self, denoise_model,emodel, x_0, t,size,c, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        x_noisy = self.q(emodel=emodel,x_0=x_0, t=t,size=size,c=c, noise=noise)
        sqrt_alphabar_t = self.extract(self.sqrt_alphabar, t, x_0.shape)
        sqrt_recip_1minalphabar = self.extract(self.sqrt_1_minalphabar, t, x_0.shape)
        actual_g = sqrt_recip_1minalphabar*(x_noisy-sqrt_alphabar_t*x_0)
        predicted_g = denoise_model(x_noisy, t)
        loss = torch.nn.functional.mse_loss(actual_g, predicted_g)

        return loss

    # -----------------------------------------------------------------------
    # Forward diffusion
    # -----------------------------------------------------------------------
    def q(self,emodel, x_0, t,size, c, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        T = 300
        k_linear = [t/T for t in range(T)]
        sqrt_alphabar_t = self.extract(self.sqrt_alphabar, t, x_0.shape)
        sqrt_one_minus_alphabar_t = self.extract(self.sqrt_one_minus_alphabar, t, x_0.shape)
        qresult = sqrt_alphabar_t * x_0 + sqrt_one_minus_alphabar_t * noise+k_linear[t.item()]*emodel(c.view(size,784)).view(x_0.shape)
        #qresult = qresult.to(self.device)
        return qresult

    # -----------------------------------------------------------------------
    # Define function allowing us to extract t index for a batch
    # -----------------------------------------------------------------------
    def extract(self, a, t, x_shape):
        '''
        a: tensor of shape (T,)
        t: time step t
        x_shape: shape of x_0
        '''
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    #---------------------------------------------
    # Sampling
    # With a trained model, we can now subtract the noise
    #---------------------------------------------
    @torch.no_grad()
    def p_sample(self,c,emodel, model, x, t, t_index):
        betas_t = self.extract(self.beta, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(self.sqrt_one_minus_alphabar, t, x.shape)
        sqrt_recip_alphas_t = self.extract(self.sqrt_1_alpha, t, x.shape)
        s_tcoef = self.extract(self.st_coef, t, x.shape)
        T = 300
        k_linear = [t/T for t in range(T)]
        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        model_mean = sqrt_recip_alphas_t * (x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t)
        coef_s_t = s_tcoef*k_linear[t_index]*emodel(c).view(x.shape)
        s_t_minusone = k_linear[t_index-1]*emodel(c).view(x.shape)
        model_mean = model_mean + coef_s_t+s_t_minusone
        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self.extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            # Algorithm 2 line 4:
            return model_mean + torch.sqrt(posterior_variance_t) * noise 

    #---------------------------------------------
    # Algorithm 2 (including returning all images)
    #---------------------------------------------
    @torch.no_grad()
    def sample(self, model,emodel,c, image_size, n_samples, channels=1):
        shape = (n_samples, channels, image_size, image_size)
        device = next(model.parameters()).device
        
        b = shape[0]
        # start from pure noise (for each example in the batch)
        img = torch.randn(shape, device=device)
        c= c.to(self.device)
        img = img.to(self.device)
        s_T = emodel(c).view(img.shape)
        x_T = img + s_T
        imgs = []

        desc = f'Generating {n_samples} samples, {self.T} time steps'
        for i in tqdm(reversed(range(0, self.T)), desc=desc, total=self.T):
            img = self.p_sample(c,emodel,model, x_T, torch.full((b,), i, device=device, dtype=torch.long), i)
            imgs.append(img.cpu().numpy())
        return imgs

# ================================================================================================
# Dataset for jet images
# ================================================================================================
class JetImageDataset(torch.utils.data.Dataset):
    def __init__(self, X,C, n_train):
        super().__init__()
        # Add a dimension for channel (expected by the model)
        X = X[:n_train,:,:]
        C = C[:n_train,:,:]
        X = np.expand_dims(X, axis=1)
        C = np.expand_dims(C, axis=1)
        self.data = torch.from_numpy(X).float()
        self.label = torch.from_numpy(C).float()
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx],self.label[idx]