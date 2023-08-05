"""
Implementation of diffusion model in pytorch to generate MNIST images.

Based on https://huggingface.co/blog/annotated-diffusion,
which implements the DDPM paper https://arxiv.org/abs/2006.11239.
"""

import torch
import random
import torchvision
import torch.nn as nn
import os
import sys
import pathlib
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm.auto import tqdm
import pickle


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

        self.results_folder = pathlib.Path(f"{self.output_dir}/ttresults")
        self.results_folder.mkdir(exist_ok = True)

        self.plot_folder = pathlib.Path(f"{self.output_dir}/ttplot")
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
        train_dataset = JetImageDataset(results['Had'],results['Cond'],
                                        self.n_train)

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
        import torch
        import torchvision
        import torch.nn as nn
        self.T = 300
        self.beta = torch.linspace(0.0001, 0.02, self.T)
        alpha = 1. - self.beta
        alphabar = torch.cumprod(alpha, axis=0)
        alphabar_prev = torch.nn.functional.pad(alphabar[:-1], (1, 0), value=1.0)
        self.sqrt_1_alpha = torch.sqrt(1.0 / alpha)

        # Quantities needed for diffusion q(x_t | x_{t-1})
        self.sqrt_alphabar = torch.sqrt(alphabar)
        self.sqrt_one_minus_alphabar = torch.sqrt(1. - alphabar)

        # Quantities needed for inversion q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.beta * (1. - alphabar_prev) / (1. - alphabar)

        # Forward diffusion example
        #imgs = next(iter(self.train_dataloader))
        #img0 = imgs[0][0]
        #difuze = self.q(img0, torch.tensor([self.T-1]))
        #plt.imshow(img0.numpy(), cmap='gray', vmin=0, vmax=1)
        #plt.savefig(f"{self.plot_folder}/img0.png")
        #plt.imshow(difuze.numpy(), cmap='gray', vmin=0, vmax=1)
        #plt.savefig(f"{self.plot_folder}/img0_diffused.png")
        #plt.clf()
        import torch



# Define the simple neural network class
        class SimpleNN(nn.Module):
            def __init__(self, input_size, output_size):
                super(SimpleNN, self).__init__()
                self.fc1 = nn.Linear(input_size, output_size)
                self.relu = nn.ReLU()
               
        
            def forward(self, x):
                x = self.fc1(x)
                x = self.relu(x)
        
                return x


    # Define the network dimensions
        input_size = 256  # Input size of the network (number of input features)
        hidden_size = 50  # Size of the hidden layer
        output_size = 128  # Output size of the network (number of classes)
    
    # Create an instance of the simple neural network
        encoder_10 = SimpleNN(256, 100)
        encoder_12 = SimpleNN(256, 144)
        decoder_10 = SimpleNN(100,256)
        decoder_10.to(self.device)
        encoder_10.to(self.device)
        encoder_12.to(self.device)

   # Creating a random input tensor of shape (1, input_size)
    

   #--------------------------------
        # Training the denoising model
        #---------------------------------------------
     
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
        opt_12 = torch.optim.Adam(encoder_12.parameters(), lr=learning_rate)
        opt_10 = torch.optim.Adam(encoder_10.parameters(), lr=learning_rate)
        dec_opt = torch.optim.Adam(decoder_10.parameters(), lr=learning_rate)
        model_outputfile = str(self.results_folder / 'model.pkl')
        if os.path.exists(model_outputfile):
            model.load_state_dict(torch.load(model_outputfile))
            print(f"Loaded trained model from: {model_outputfile} (delete and re-run if you'd like to re-train)")
        else:
            training_loss = []
            dtraining_loss = []
            for epoch in range(n_epochs):
                print(f'Epoch {epoch}')
                for step, (H,P) in enumerate(self.train_dataloader):
                
                    H = H.to(self.device)
                    P = P.to(self.device)
                    # Algorithm 1 line 3: sample t uniformally for every example in the batch
                    t = torch.randint(0, self.T, (self.batch_size,), device=self.device).long()
                    loss = self.p_losses(model,encoder_10,encoder_12, H,P, t)
                    training_loss.append(loss.cpu().detach().numpy().item())
                    Pencodeddecoded = decoder_10(encoder_10(P.view(P.shape[0],256))).view(P.shape[0],1,16,16)
                    loss_d =torch.nn.functional.mse_loss(P, Pencodeddecoded)
                    dtraining_loss.append(loss_d.cpu().detach().numpy().item())
                    if step % 100 == 0:
                        print(f"  Loss (step {step}):", loss.item())
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    opt_12.step()
                    opt_12.zero_grad()
                    opt_10.step()
                    opt_10.zero_grad()
                    loss_d.backward()
                    dec_opt.step()
                    dec_opt.zero_grad()


            print('Done training!')
            torch.save(model.state_dict(), model_outputfile)
            print(f'Saved model: {model_outputfile}')
            plt.plot(training_loss)
            plt.xlabel('Training step')
            plt.ylabel('Loss')
            plt.savefig(str(self.plot_folder / f'loss.png'))
            plt.clf()
            plt.plot(dtraining_loss)
            plt.xlabel('Training step')
            plt.ylabel('Decoder Loss')
            plt.savefig(str(self.plot_folder / f'dloss.png'))
            plt.clf()

        #---------------------------------------------
        # Sampling from the trained model
        #Create test conditions 
        print('Creating test samples')
        # Define Hadronization function
        im_dim = 16
        num_samples = 1000
        def choose_z():
            z =[]
            N = random.randint(2,6)
            for k in range(N):
                if k == 0:
                    z_0 = random.uniform(0, 0.9)
                    z.append(z_0)
                elif k == N-1:
                    z.append(1-sum(z))
                else:
                    z.append(random.uniform(0, 1-sum(z)))
            return z
        def hadronize(P):
            L =[]
            while len(L)<2:
                for i in range(im_dim):
                    for j in range(im_dim):
                        if P[i,j]>0:
                            M = np.zeros((im_dim, im_dim))
                            z = choose_z()
                            for l in range(len(z)):
                                eta = np.round(np.random.normal(i, 1, 1)[0]).astype(np.int32)  
                                phi = np.round(np.random.normal(j, 1, 1)[0]).astype(np.int32)  
                                if (eta in range(im_dim) and phi in range(im_dim)):
                                    M[eta,phi] = z[l]
                            L.append(M) 
            return sum(L)
#Define two different looking starting jets
        Point = np.zeros((im_dim, im_dim))
        Point[6,6] = 1
#Define two different looking starting jets
        Square = np.zeros((im_dim, im_dim), dtype=int)
        index =[i+int(im_dim/4) for i in range(int(im_dim/2)) if i%2 ==0 ]
        for i in range(len(index)):
            Square[index[i],int(im_dim/4)] = 1
            Square[int(im_dim/4),index[i]] = 1
            Square[index[i],int(im_dim*3/4)] = 1
            Square[int(im_dim*3/4),index[i]] = 1

#Create data set 
        P_list = [[hadronize(Point),Point] for i in range(num_samples)]
        S_list = [[hadronize(Square),Square] for i in range(num_samples)]
        Sample_list = P_list + S_list
        Shufflehad_list = []
        Conditions_list = []
        for i in range(num_samples):
            x = random.randint(0,len(Sample_list)-1)
            Shufflehad_list.append(Sample_list[x][0])
            Conditions_list.append(Sample_list[x][1])
        Hadronsarray = np.stack(Shufflehad_list)
        Consarray = np.stack(Conditions_list)
        Hadronsarray= torch.tensor(Hadronsarray).to(self.device)
        Hadronsarray = Hadronsarray.to(torch.float32)
        print(Hadronsarray.shape,Consarray.shape)

      
       #------------------------------
        print()
        print('------------------ Sampling ------------------')
        print('--------------------------------------------')
      
        n_samples = 1000
        samples_outputfile = str(self.results_folder / 'samples.pkl')
        if os.path.exists(samples_outputfile):
            with open(samples_outputfile, "rb") as f:
                samples = pickle.load(f) 
            print(f"Loaded samples from: {samples_outputfile} (delete and re-run if you'd like to re-train)")
        else:
            samples = self.sample(model,encoder_10,encoder_12,decoder_10,Hadronsarray, image_size=self.image_dim, n_samples=n_samples)

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
        #train_dataset = self.train_dataloader.dataset
        #indices = torch.randperm(len(train_dataset))[:n_samples]
    

        # Now, samples_0 and train_samples are of shape (n_samples, image_dim, image_dim)

        # N pixels above threshold
        #threshold = 1.e-2
        #N_generated = np.sum(samples_0 > threshold, axis=(1,2))
        #N_train = np.sum(train_samples > threshold, axis=(1,2))
        
    #plot_results.plot_histogram_1d(x_list=[N_generated, N_train], 
     #                                  label_list=['generated', 'target'],
      #                                 bins=np.linspace(-0.5, 29.5, 31),
       #                                xlabel=f'N pixels with z>{threshold}', 
        #                               filename='N_pixels.png', 
         #                              output_dir=self.plot_folder)

        # z distribution
        #z_generated = samples_0.flatten()
        #z_train = train_samples.flatten()
        #plot_results.plot_histogram_1d(x_list=[z_generated, z_train], 
         #                              label_list=['generated', 'target'],
          #                             bins=np.linspace(0., 1., 101),
           #                            logy=True,
            #                           xlabel=f'z (pixels)', 
             #                          filename='z.png', 
              #                         output_dir=self.plot_folder)

        # Plot some sample images
        for random_index in range(5):

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
            plt.imshow(Consarray[random_index].reshape(self.image_dim, self.image_dim, 1), cmap="gray")
            plt.savefig(str(self.plot_folder / f'{random_index}_expected.png'))
            plt.clf()
        sys.exit()    

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
    def p_losses(self, denoise_model,encoder_10,encoder_12, H,P, t, noise=None):
        batch_size = P.shape[0]
        if noise is None:
            noise = torch.randn(batch_size,10,10).to(self.device)
        Pencoded = encoder_10(P.view(batch_size,256)).view(batch_size,10,10)
        x_noisy = self.q(x_0=Pencoded, t=t, noise=noise)
        zeros =torch.zeros(batch_size,12).to(self.device)
        x_noisy = x_noisy.view(batch_size,100)
        Hencoded = encoder_12(H.view(H.shape[0],256))
        merged_tensor = torch.cat((x_noisy, zeros), dim=1)
        merged_tensor_1 = torch.cat((merged_tensor, Hencoded), dim=1).view(batch_size,1,16,16)
        u_output = denoise_model(merged_tensor_1, t).view(batch_size, 256)
        predicted_noise = encoder_10(u_output).view(batch_size,10,10)
        loss = torch.nn.functional.mse_loss(noise, predicted_noise)

        return loss

    # -----------------------------------------------------------------------
    # Forward diffusion
    # -----------------------------------------------------------------------
    def q(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alphabar_t = self.extract(self.sqrt_alphabar, t, x_0.shape)
        sqrt_one_minus_alphabar_t = self.extract(self.sqrt_one_minus_alphabar, t, x_0.shape)

        return sqrt_alphabar_t * x_0 + sqrt_one_minus_alphabar_t * noise

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
    def p_sample(self, model,encoder_10,encoder_12,H, x, t, t_index):
        batch_size = H.shape[0]
        betas_t = self.extract(self.beta, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(self.sqrt_one_minus_alphabar, t, x.shape)
        sqrt_recip_alphas_t = self.extract(self.sqrt_1_alpha, t, x.shape)
        #Reshape to input in the U-net
        x_flat = x.view(batch_size,100)
        zeros =torch.zeros(batch_size,12).to(self.device)
        Hencoded = encoder_12(H.view(H.shape[0],256))
        merged_tensor = torch.cat((x_flat, zeros), dim=1)
        merged_tensor_1 = torch.cat((merged_tensor, Hencoded), dim=1).view(batch_size,1,16,16)
        u_output = model(merged_tensor_1, t).view(batch_size, 256)
        e_noise = encoder_10(u_output).view(batch_size, 10,10)
    
        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        model_mean = sqrt_recip_alphas_t * (x - betas_t * e_noise / sqrt_one_minus_alphas_cumprod_t)

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
    def sample(self, model,encoder_10,encoder_12,decoder,H, image_size, n_samples, channels=1):
        shape = (n_samples, channels, image_size, image_size)
        device = next(model.parameters()).device
        b = shape[0]
        # start from pure noise (for each example in the batch)
        img = torch.randn(H.shape[0],10,10).to(self.device)
        imgs = []
        decimgs = []
        desc = f'Generating {n_samples} samples, {self.T} time steps'
        for i in tqdm(reversed(range(0, self.T)), desc=desc, total=self.T):
            img = self.p_sample(model,encoder_10,encoder_12,H, img, torch.full((b,), i, device=device, dtype=torch.long), i)
            imgs.append(img)
        for j in range(len(imgs)):
            dec = decoder(imgs[j].view(imgs[j].shape[0],100)).view(imgs[j].shape[0],16,16)
            decimgs.append(dec.cpu().numpy())
        return decimgs

# ================================================================================================
# Dataset for jet images
# ================================================================================================
class JetImageDataset(torch.utils.data.Dataset):
    def __init__(self, H,C, n_train):
        super().__init__()
        # Add a dimension for channel (expected by the model)
        H = H[:n_train,:,:]
        H = np.expand_dims(H, axis=1)
        self.data = torch.from_numpy(H).float()
        C = C[:n_train,:,:]
        C = np.expand_dims(C, axis=1)
        self.c = torch.from_numpy(C).float()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx],self.c[idx]