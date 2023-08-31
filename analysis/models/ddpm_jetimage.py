"""
Implementation of diffusion model in pytorch to generate MNIST images.

Based on https://huggingface.co/blog/annotated-diffusion,
which implements the DDPM paper https://arxiv.org/abs/2006.11239.
"""

import os
import sys
import random
import pathlib
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm.auto import tqdm
import pickle
from matplotlib.backends.backend_pdf import PdfPages
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms

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

        self.results_folder = pathlib.Path(f"{self.output_dir}/forwardhadresults")
        self.results_folder.mkdir(exist_ok = True)

        self.plot_folder = pathlib.Path(f"{self.output_dir}/forwardhadplot")
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
        train_dataset = JetImageDataset(results['Had'],results['Part'],
                                        self.n_train)
        self.conditions = results['Part']
        self.had = results['Had']
        self.split_forward_sampling = True
        self.reverse_hadronization = False
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
        self.T = 1000
        self.beta = torch.linspace(0.0001, 0.02, self.T)
        alpha = 1. - self.beta
        alphabar = torch.cumprod(alpha, axis=0)
        alphabar_prev = torch.nn.functional.pad(alphabar[:-1], (1, 0), value=1.0)
        self.sqrt_1_alpha = torch.sqrt(1.0 / alpha)
        self.kt = torch.linspace(0.0, 1.0, self.T)
        self.kt_prev = torch.nn.functional.pad(self.kt[:-1], (1, 0), value=1.0)
        # Quantities needed for diffusion q(x_t | x_{t-1})
        self.sqrt_alphabar = torch.sqrt(alphabar)
        self.sqrt_one_minus_alphabar = torch.sqrt(1. - alphabar)
        self.oneoversqrt_one_minus_alphabar = torch.sqrt(1./(1. - alphabar))

        # Quantities needed for inversion q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.beta * (1. - alphabar_prev) / (1. - alphabar)
        self.stcoef = -torch.sqrt(alpha)*(1.0-alphabar_prev)/(1.-alphabar)
        # Forward diffusion example
        

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
        #U-net for g

        class Block(nn.Module):
            def __init__(self, in_ch, out_ch):
                super().__init__()
                self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
                self.relu = nn.ReLU()
                self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

            def forward(self, x):
                return self.conv2(self.relu(self.conv1(x)))

        class Encoder(nn.Module):
            def __init__(self, chs=(1, 64, 128, 256, 512)):
                super().__init__()
                self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)])
                self.pool = nn.MaxPool2d(2)
        
            def forward(self, x):
                ftrs = []
                for block in self.enc_blocks:
                    x = block(x)
                    ftrs.append(x)
                    if x.shape[2] > 4:  # Prevent dimensions from becoming too small
                        x = self.pool(x)
                return ftrs


        class Decoder(nn.Module):
            def __init__(self, chs=(1024, 512, 256, 128, 64)):
                super().__init__()
                self.chs = chs
                self.upconvs = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i + 1], 2, 2) for i in range(len(chs) - 1)])
                self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)])

            def forward(self, x, encoder_features):
                for i in range(len(self.chs) - 1):
                    x = self.upconvs[i](x)
                    enc_ftrs = self.crop(encoder_features[i], x)
                    x = torch.cat([x, enc_ftrs], dim=1)
                    x = self.dec_blocks[i](x)
                return x

            def crop(self, enc_ftrs, x):
                _, _, H, W = x.shape
                enc_ftrs = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
                return enc_ftrs

        class UNet(nn.Module):
            def __init__(self, enc_chs=(1, 64, 128, 256, 512, 1024), dec_chs=(1024, 512, 256, 128, 64), num_class=1, retain_dim=False, out_sz=(572, 572)):
                super().__init__()
                self.encoder = Encoder(enc_chs)
                self.decoder = Decoder(dec_chs)
                self.head = nn.Conv2d(dec_chs[-1], num_class, 1)
                self.retain_dim = retain_dim
                self.out_sz = out_sz

            def forward(self, x):
                enc_ftrs = self.encoder(x)
                out = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
                out = self.head(out)
                if self.retain_dim:
                    out = F.interpolate(out, size=self.out_sz, mode='bilinear', align_corners=False)
                return out

# Create an instance of the UNet model
        input_tensor = torch.randn(1, 1, 16, 16)  # 16x16 single-channel input image
        Emodel = UNet(num_class=1, retain_dim=True, out_sz=(16, 16))
        output = Emodel(input_tensor)
        Emodel.to(self.device)
        print("Output shape:", output.shape)
       
        print()
        print('------------------ Training ------------------')
        # Hyperparameters
        learning_rate = self.model_params['learning_rate']
        n_epochs = self.model_params['n_epochs']
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        opt = torch.optim.Adam(Emodel.parameters(), lr=learning_rate)
        model_outputfile = str(self.results_folder / 'model.pkl')
        if os.path.exists(model_outputfile):
            model.load_state_dict(torch.load(model_outputfile))
            print(f"Loaded trained model from: {model_outputfile} (delete and re-run if you'd like to re-train)")
        else:
            training_loss = []
            for epoch in range(n_epochs):
                print(f'Epoch {epoch}')
                for step, (P,H) in enumerate(self.train_dataloader):
                
                    P = P.to(self.device)
                    H = H.to(self.device)
                    # Algorithm 1 line 3: sample t uniformally for every example in the batch
                    t = torch.randint(0, self.T, (self.batch_size,), device=self.device).long()

                    loss = self.p_losses(model,Emodel, P,H, t)
                    training_loss.append(loss.cpu().detach().numpy().item())

                    if step % 100 == 0:
                        print(f"  Loss (step {step}):", loss.item())

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    opt.step()
                    opt.zero_grad()

            print('Done training!')
            torch.save(model.state_dict(), model_outputfile)
            print(f'Saved model: {model_outputfile}')
            plt.plot(training_loss)
            plt.xlabel('Training step')
            plt.ylabel('Loss')
            plt.savefig(str(self.plot_folder / f'loss.png'))
            plt.clf()
            print('training works')
            
        #---------------------------------------------
        # Sampling from the trained model
        #---------------------------------------------
        print()
        print('------------------ Sampling ------------------')
        print('--------------------------------------------')
        n_samples = 600
        C = self.conditions[:n_samples,:,:]
        
        
        C = torch.from_numpy(C).unsqueeze(1).float().to('cpu')  
        Emodel.to('cpu')
        s_T = Emodel(C)
        s_T = s_T.to(self.device)
        samples_outputfile = str(self.results_folder / 'samples.pkl')
        if os.path.exists(samples_outputfile):
            with open(samples_outputfile, "rb") as f:
                samples = pickle.load(f) 
            print(f"Loaded samples from: {samples_outputfile} (delete and re-run if you'd like to re-train)")
        else:
            samples = self.sample(model,s_T, image_size=self.image_dim, n_samples=n_samples)

            with open(samples_outputfile, "wb") as f:
                pickle.dump(samples, f)
            print(f'Saved {n_samples} samples: {samples_outputfile}')
        

        # Get the generated images (i.e. last time step)
        samples_0 = np.squeeze(samples[self.T-2])
        if self.split_forward_sampling:
            n_samples = 10000
            C = self.conditions[:n_samples,:,:]
            H = self.had[:n_samples,:,:]
            A_jet = C[0]
            B_jet = A_jet
            j = 1
            while np.linalg.norm(A_jet - B_jet) == 0:
                B_jet = C[j]
                j = j+1
            plt.imshow(A_jet, cmap="gray")
            plt.savefig(str(self.plot_folder / 'A_jet.png'))
            plt.clf()
            plt.imshow(B_jet, cmap="gray")
            plt.savefig(str(self.plot_folder / 'B_jet.png'))
            plt.clf()
            
            A_index = [i for i in range(n_samples) if np.linalg.norm(A_jet - C[i]) == 0]
            B_index = [i for i in range(n_samples) if np.linalg.norm(B_jet - C[i]) == 0]
            print(len(A_index),len(B_index),len(A_index)+len(B_index))
            CA_list = [C[k] for k in A_index]
            HA_list = [H[k] for k in A_index]
            CB_list = [C[k] for k in B_index]
            HB_list = [H[k] for k in B_index]
            CA = np.stack(CA_list)
            HA = np.stack(HA_list)
            CB = np.stack(CB_list)
            HB = np.stack(HB_list)
            CA = torch.from_numpy(CA).unsqueeze(1).float().to('cpu')  
            Emodel.to('cpu')
            s_T = Emodel(CA)
            s_T = s_T.to(self.device)
            Asamples = self.sample(model,s_T, image_size=self.image_dim, n_samples=len(CA_list))
            Asamples_0 = np.squeeze(Asamples[self.T-2])
            # z_A distribution
            z_A_generated = Asamples_0.flatten()
            z_A_train = HA.flatten()
            plot_results.plot_histogram_1d(x_list=[z_A_generated, z_A_train], 
                                       label_list=['generated', 'target'],
                                       bins=np.linspace(0., 1., 100),
                                       logy=True,
                                       xlabel=f'z of jet A(pixels)', 
                                       filename='z_A.png', 
                                       output_dir=self.plot_folder)
            # N pixels above threshold
            threshold = 0.001
            N_A_generated = np.sum(Asamples_0, axis=(1,2))
            N_A_train = np.sum(HA, axis=(1,2))
            plot_results.plot_histogram_1d(x_list=[N_A_generated, N_A_train], 
                                       label_list=['generated', 'target'],
                                       bins=np.linspace(-0.5, 29.5, 31),
                                       xlabel=f'N pixels of jets generated from jet A', 
                                       filename='N_A_pixels.png', 
                                       output_dir=self.plot_folder)
            print('Done with jet A')
            #Now B
            
            CB = torch.from_numpy(CB).unsqueeze(1).float().to('cpu')  
            Emodel.to('cpu')
            s_T = Emodel(CB)
            s_T = s_T.to(self.device)
            Bsamples = self.sample(model,s_T, image_size=self.image_dim, n_samples=len(CB_list))
            Bsamples_0 = np.squeeze(Bsamples[self.T-2])
            # z_B distribution
            z_B_generated = Bsamples_0.flatten()
            z_B_train = HB.flatten()
            plot_results.plot_histogram_1d(x_list=[z_B_generated, z_B_train], 
                                       label_list=['generated', 'target'],
                                       bins=np.linspace(0., 1., 100),
                                       logy=True,
                                       xlabel=f'z of jet B(pixels)', 
                                       filename='z_B.png', 
                                       output_dir=self.plot_folder)
            # N pixels above threshold
            threshold = 0.001
            N_generated = np.sum(Bsamples_0, axis=(1,2))
            N_train = np.sum(HB , axis=(1,2))
            plot_results.plot_histogram_1d(x_list=[N_generated, N_train], 
                                       label_list=['generated', 'target'],
                                       bins=np.linspace(-0.5, 29.5, 31),
                                       xlabel=f'N pixels with z of jets generated from jet B', 
                                       filename='N_B_pixels.png', 
                                       output_dir=self.plot_folder)

            
        print('successful sampling')
        #---------------------------------------------
        # Plot some observables
        #---------------------------------------------  
        # Plot some sample images
        print()
        print('------------------ Test Accuracy (in case you are running reverse hadronization)------------------')
        print('--------------------------------------------')
        if self.reverse_hadronization:
            C = C.to('cpu').numpy()
            C = self.conditions[:n_samples,:,:]
            H = self.had[:n_samples,:,:]
            good_predictions = 0
            accuracy_test = 100
            jet_A= C[0].reshape(self.image_dim, self.image_dim, 1)
            jet_B= C[1].reshape(self.image_dim, self.image_dim, 1)



            plt.imshow(jet_A, cmap="gray")
            plt.savefig(str(self.plot_folder / 'FjetA.png'))
            plt.clf()
            plt.imshow(jet_B, cmap="gray")
            plt.savefig(str(self.plot_folder / 'FjetB.png'))
            plt.clf()
            true_negative = 0
            false_positive = 0
            false_negative = 0
            true_positive = 0
            num_ajets = 0
            num_bjets = 0
            print('number of samples available',samples_0.shape[0] )
            for random_index in range(accuracy_test):
                index = np.random.randint(low=0, high=samples_0.shape[0]-1)
                expected = H[index].reshape(self.image_dim, self.image_dim, 1)
                condition = C[index].reshape(self.image_dim, self.image_dim, 1)
                generated = samples[998][index].reshape(self.image_dim, self.image_dim, 1)
                distance= np.linalg.norm(expected - generated)
                if np.linalg.norm(condition - jet_A)==0:
                    num_ajets = num_ajets +1
                elif np.linalg.norm(condition - jet_B)==0:
                    num_bjets = num_bjets +1
                if distance<0.5:
                    good_predictions = good_predictions+1
                    if np.linalg.norm(expected - jet_A)==0:
                        true_negative = true_negative+1
                    else:
                        true_positive = true_positive+1
                else:
                    if np.linalg.norm(expected - jet_A)==0:
                        false_negative = false_negative+1
                    else:
                        false_positive = false_positive+1
        

            print('Total good predictions: ', good_predictions,'Out of: ',accuracy_test)
            print('Accuracy: ',good_predictions/accuracy_test*100,'%' )
            print('True negative: ', true_negative)
            print('False positive: ', false_positive)
            print('TN+FN: ', true_negative+false_negative)
            print('number of A jets', num_ajets)
            print('False negative: ', false_negative)
            print('True positive: ', true_positive)
            print('FP+TP: ', false_positive+true_positive)
            print('number of B jets', num_bjets)
            print()
            print('True Positive rate: ', true_positive/(true_positive+false_negative))
            print('False Positive rate: ', false_positive/(false_positive+true_negative))
            fig, ax = plt.subplots()

# Plot the plane
            model_point = (false_positive/(false_positive+true_negative),true_positive/(true_positive+false_negative))
            plt.plot([0, 1], [0, 1], color='blue')
            plt.scatter(*model_point, color='red', label='Shift-DDPM reverse hadronization model')
            ax.annotate('Shift-DDPM reverse hadronization model', xy=model_point, xytext=(model_point[0], model_point[1] - 0.1),
            color='red', arrowprops=dict(arrowstyle='->', color='red'))
# Set axis titles
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            plt.xlim(0, 1.01)        
            plt.ylim(0, 1.01)
            plt.savefig(str(self.plot_folder / 'Fmodelroc.png'))
      








        print('------------------ Plotting ------------------')
        print('--------------------------------------------')

        # Get images from the training set, for comparison
        #train_dataset = self.train_dataloader.dataset.data
        #indices = torch.randperm(len(train_dataset))[:n_samples]
        #train_samples = np.squeeze(torch.stack([train_dataset[idx] for idx in indices]).numpy())

        # Now, samples_0 and train_samples are of shape (n_samples, image_dim, image_dim)

        # N pixels above threshold
        #threshold = 0.001
        #N_generated = np.sum(samples_0, axis=(1,2))
        #N_train = np.sum(train_samples > threshold, axis=(1,2))
        #plot_results.plot_histogram_1d(x_list=[N_generated, N_train], 
         #                              label_list=['generated', 'target'],
          #                             bins=np.linspace(-0.5, 29.5, 31),
           #                            xlabel=f'N pixels with z>{threshold}', 
            #                           filename='N_pixels.png', 
             #                          output_dir=self.plot_folder)

        # z distribution
        #z_generated = samples_0.flatten()
        #z_train = train_samples.flatten()
        #plot_results.plot_histogram_1d(x_list=[z_generated, z_train], 
         #                              label_list=['generated', 'target'],
          #                             bins=np.linspace(0., 1., 100),
           #                            logy=True,
            #                           xlabel=f'z (pixels)', 
             #                          filename='z.png', 
              #                         output_dir=self.plot_folder)

        # Plot some sample images
        #C = C.to('cpu').numpy()
        #C = self.conditions[:n_samples,:,:]
        #H = self.had[:n_samples,:,:]
        #for random_index in range(10):
         #   plt.clf()
          #  plt.imshow(samples[998][random_index].reshape(self.image_dim, self.image_dim, 1), cmap="gray")
           # plt.savefig(str(self.plot_folder / f'{random_index}_generated.png'))
           # plt.clf()
            
           # plt.imshow(C[random_index].reshape(self.image_dim, self.image_dim, 1), cmap="gray")
           # plt.savefig(str(self.plot_folder / f'{random_index}_input.png'))
           # plt.clf()
           # plt.imshow(H[random_index].reshape(self.image_dim, self.image_dim, 1), cmap="gray")
           # plt.savefig(str(self.plot_folder / f'{random_index}_expected.png'))
           # plt.clf()

            # Generate a gif of denoising
        #for j in range(3):
         #   k = random.randint(1, 100)
           # fig = plt.figure()
           # ims = []
           # for i in range(1000):
            #    im = plt.imshow(samples[i][random_index].reshape(self.image_dim, self.image_dim, 1), cmap="gray", animated=True)
             #   ims.append([im])
            
            #animate = animation.ArtistAnimation(fig, ims, interval=2, blit=True, repeat_delay=9000)
            #animate.save(str(self.plot_folder / f'{random_index}_generated.gif'))
            # Get the last frame from ims
            # Create a new figure for the last frame
            #fig_last = plt.figure()
            #ax_last = fig_last.add_subplot(111)

# Get the last frame from ims
            #last_frame = ims[-2][0].get_array()  # Get the image data from the AxesImage object

# Display the last frame
            #ax_last.imshow(last_frame, cmap="gray")

# Save the last frame as a PDF
            #pdf_filename = str(self.plot_folder / f'{random_index}_slast_frame.pdf')
            #pp = PdfPages(pdf_filename)
            #pp.savefig(fig_last)
            #pp.close()

      
        #Sample images
        #n_samples = 10
        #for i in range(n_samples):
         #   j = random.randint(1, 100)
          #  C = self.conditions[j]
           # C = torch.from_numpy(C).unsqueeze(1).float().to('cpu')
           # C= C.view(1,1,16,16)
           # H =  self.had[j]
           # Emodel.to('cpu')
           # s_T = Emodel(C)
           # s_T = s_T.to(self.device)       
           # ssamples = self.sample(model,s_T, image_size=self.image_dim, n_samples=1)
           # ssamples_0 = np.squeeze(ssamples[self.T-1])
           # plt.imshow(ssamples_0.reshape(self.image_dim, self.image_dim, 1), cmap="gray")
           # plt.savefig(str(self.plot_folder / f'{i}_generated.png'))
           # plt.clf()
           # plt.imshow(C.reshape(self.image_dim, self.image_dim, 1), cmap="gray")
           # plt.savefig(str(self.plot_folder / f'{i}_condition.png'))
           # plt.clf()
           # plt.imshow(H.reshape(self.image_dim, self.image_dim, 1), cmap="gray")
           # plt.savefig(str(self.plot_folder / f'{i}_expected.png'))
           # plt.clf()
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

    def p_losses(self, denoise_model,emodel, x_0,C, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        
        kt = self.extract(self.kt, t, x_0.shape)
        s_t = kt*emodel(C)
        x_noisy = self.q(x_0=x_0, t=t, noise=noise)+s_t
        
        sqrt_alphabar_t = self.extract(self.sqrt_alphabar, t, x_0.shape)
        oneover = self.extract(self.oneoversqrt_one_minus_alphabar, t, x_0.shape)
        expected_g = oneover*(x_noisy-sqrt_alphabar_t*x_0)
        predicted_g = denoise_model(x_noisy, t)
    
        loss = torch.nn.functional.mse_loss(expected_g, predicted_g)

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
    def p_sample(self, model,s_T, x, t, t_index):
        kt =self.extract(self.kt, t, x.shape)
        kt_prev = self.extract(self.kt_prev, t, x.shape)
        betas_t = self.extract(self.beta, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(self.sqrt_one_minus_alphabar, t, x.shape)
        sqrt_recip_alphas_t = self.extract(self.sqrt_1_alpha, t, x.shape)
        stcoef = self.extract(self.stcoef, t, x.shape)
        
        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        model_mean = sqrt_recip_alphas_t * (x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t)
        model_mean = model_mean+stcoef*(kt*s_T)+kt_prev*s_T
       
        if (t_index == 0 or t_index ==1):
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

    def sample(self, model,s_T, image_size, n_samples, channels=1):
        
        shape = (n_samples, channels, image_size, image_size)
        device = next(model.parameters()).device
        b = shape[0]

        # start from pure noise (for each example in the batch)
        img = torch.randn(shape, device=device)+s_T
        imgs = []

        desc = f'Generating {n_samples} samples, {self.T} time steps'
        for i in tqdm(reversed(range(0, self.T)), desc=desc, total=self.T):
            img = self.p_sample(model,s_T, img, torch.full((b,), i, device=device, dtype=torch.long), i)
            imgs.append(img.cpu().numpy())
        return imgs

# ================================================================================================
# Dataset for jet images
# ================================================================================================
class JetImageDataset(torch.utils.data.Dataset):
    def __init__(self, P,H, n_train):
        super().__init__()
        # Add a dimension for channel (expected by the model)
        P = P[:n_train,:,:]
        P = np.expand_dims(P, axis=1)
        self.data = torch.from_numpy(P).float()
        H = H[:n_train,:,:]
        H = np.expand_dims(H, axis=1)
        self.had = torch.from_numpy(H).float()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.had[idx]