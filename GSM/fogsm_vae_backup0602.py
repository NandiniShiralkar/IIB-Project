from _imports import *
from torch.distributions.multivariate_normal import MultivariateNormal
from scipy.linalg import cholesky, solve_triangular
from torch.nn import Parameter
from tqdm import tqdm
from memory_profiler import profile


class FoGSMModel(nn.Module):
    def __init__(self,thetas=None, length_scale_feature=1.0, length_scale_amplitude=1.2, kappa=1.0, jitter=1e-5, grid_size=10, frequency=1.0,sigma=0.1):
        super(FoGSMModel, self).__init__()

        self.dtype = torch.float64

        self.length_scale_feature = Parameter(torch.tensor(length_scale_feature, dtype=self.dtype))
        self.length_scale_amplitude = Parameter(torch.tensor(length_scale_amplitude, dtype=self.dtype))
        self.kappa = Parameter(torch.tensor(kappa, dtype=self.dtype))
        self.frequency = Parameter(torch.tensor(frequency, dtype=self.dtype)) 
        self.sigma = Parameter(torch.tensor(sigma))

        if thetas is None:
            thetas = torch.linspace(0, 2 * np.pi, 8)  # 8 orientations from 0 to 2*pi
        self.thetas = thetas

        self.jitter = jitter
        self.grid_size = grid_size
        self.grid = torch.stack(torch.meshgrid(torch.linspace(-5, 5, grid_size), 
                                               torch.linspace(-5, 5, grid_size)), 
                                dim=-1).reshape(-1, 2)
        
        self.K_g = self.generate_K_g()
        
    def von_mises_kernel(self, theta1, theta2):
        theta_diff = theta1 - theta2  
        return torch.exp(self.kappa * torch.cos(theta_diff))
    
    def squared_exponential_kernel(self, x1, x2, length_scale,jitter="True"):
        x1 = x1.unsqueeze(1) # Shape: [N, 1, 2]
        x2 = x2.unsqueeze(0) # Shape: [1, N, 2]
        sq_dist = torch.sum((x1 - x2) ** 2, dim=2) # Shape: [N, N]

        exp_term = torch.exp(-sq_dist / (2*length_scale**2))

        if jitter:
            return exp_term + self.jitter * torch.eye(x1.size(0))
        else:
            return exp_term

    def composite_feature_kernel(self, theta1, theta2):
         
        sq_exp_component = self.squared_exponential_kernel(self.grid, self.grid, self.length_scale_feature,jitter="False")        
        
        # Ensure theta1 and theta2 are tensors
        theta1 = torch.tensor(theta1)
        theta2 = torch.tensor(theta2)
        x1 = self.grid.unsqueeze(1) # Shape: [N, 1, 2]
        x2 = self.grid.unsqueeze(0) # Shape: [1, N, 2]

        n1 = torch.tensor([torch.cos(theta1), torch.sin(theta1)]).view(1, 1, 2)  # Shape: [1, 1, 2]
        n2 = torch.tensor([torch.cos(theta2), torch.sin(theta2)]).view(1, 1, 2)  # Shape: [1, 1, 2]
        average_orientation = (n1 + n2) / 2

        # Broadcasting average_orientation for dot product computation
        average_orientation = average_orientation.repeat(x1.size(0), x1.size(1), 1)
        dot_product = torch.sum((x1 - x2) * average_orientation, dim=2)
        periodic_component = torch.cos(2 * torch.pi * self.frequency * dot_product)

        # Composite Kernel
        return sq_exp_component * periodic_component

    def generate_K_g(self):
        
        theta1_grid, theta2_grid = torch.meshgrid(self.thetas, self.thetas)
        ori_kernel_val = self.von_mises_kernel(theta1_grid, theta2_grid)
    
        # Spatial kernel
        loc_kernel_val = torch.zeros((len(self.thetas), len(self.thetas), self.grid_size**2, self.grid_size**2))

        for i in range(len(self.thetas)):
            for j in range(len(self.thetas)):
                loc_kernel_val[i,j] = self.composite_feature_kernel(self.thetas[i], self.thetas[j])
        
        K_spatial = torch.sum(loc_kernel_val, dim=[0, 1])
        K_g = torch.kron(K_spatial, ori_kernel_val)
        
        return K_g + self.jitter * torch.eye(len(self.thetas)*self.grid_size**2)

    def compute_A(self):
        kernel_vals = self.squared_exponential_kernel(self.grid, self.grid, self.length_scale_amplitude)
        return torch.sqrt(torch.exp(MultivariateNormal(torch.zeros(self.grid.size(0)), kernel_vals).sample()))

    def samples(self):

        g = MultivariateNormal(torch.zeros(len(self.thetas)*(self.grid_size**2)), self.K_g).sample()  

        A = self.compute_A()
        
        # Tile amplitudes to match feature fields 
        A = A.repeat(len(self.thetas))
    
        # Combine
        I = g * A  + torch.randn_like(g) * self.sigma
        I = torch.sum(I.reshape(len(self.thetas), self.grid_size, self.grid_size), dim=0)

        return I, g
    
    def visualise(self, combined_fields):

        # Normalise the combined image for visualisation
        combined_fields_normalised = combined_fields / combined_fields.max()

        # Reshape to image format
        combined_image = combined_fields_normalised.view(self.grid_size, self.grid_size).detach().numpy()

        # Visualise the combined image
        plt.figure(figsize=(5,5))
        plt.imshow(combined_image, cmap='gray') 
        plt.title('FoGSM Sample')
        plt.axis("off")
        plt.show()
   
class VariationalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, g_dim, A_dim):
        super(VariationalEncoder, self).__init__()

        self.fc_layers = nn.ModuleList([nn.Linear(input_dim, hidden_dims[0]), nn.ReLU()])

        for i in range(len(hidden_dims)):
            self.fc_layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            self.fc_layers.append(nn.ReLU())
        
        self.fc_g_mean = nn.Linear(hidden_dims[-1], g_dim)
        # Adjust for Cholesky parameter output size: g_dim * (g_dim + 1) // 2 for the lower triangular matrix
        self.fc_g_chol = nn.Linear(hidden_dims[-1], g_dim * (g_dim + 1) // 2)

        self.fc_A_mean = nn.Linear(hidden_dims[-1], A_dim)
        self.fc_A_logvar = nn.Linear(hidden_dims[-1], A_dim)
        
    def forward(self, x):
        for layer in self.fc_layers:
            x = layer(x)
        g_mean = self.fc_g_mean(x)
        g_chol_flat = self.fc_g_chol(x)        
        A_mean = self.fc_A_mean(x)
        A_logvar = self.fc_A_logvar(x)
        return g_mean, g_chol_flat, A_mean, A_logvar

class VariationalDecoder(nn.Module):
    def __init__(self, output_dim, hidden_dims, g_dim, A_dim):
        super(VariationalDecoder, self).__init__()

        input_dim = g_dim + A_dim
        self.fc_layers = nn.ModuleList()
        
        # Start the first layer with the combined size of g and A
        self.fc_layers.append(nn.Linear(input_dim, hidden_dims[0]))
        self.fc_layers.append(nn.ReLU())
        
        for i in range(1, len(hidden_dims)):
            self.fc_layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
            self.fc_layers.append(nn.ReLU())
        
        # Output layer to reconstruct the original input dimensions
        self.fc_output = nn.Linear(hidden_dims[-1], output_dim)
    
    def forward(self, x):
        for layer in self.fc_layers:
            x = layer(x)
        reconstructed_x = self.fc_output(x)
        return reconstructed_x


class FoGSMVAE(FoGSMModel):

    """ 
    
    Variational Autoencoder for FoGSM

        - reparameterise: Samples from the distribution defined by mean and logvar multiple times.
        - forward: Performs a forward pass through the VAE.
        - compute_reconstruction_loss: Computes the reconstruction loss part of the ELBO.
        - compute_kl_divergence: Computes the KL divergence between the approximate and the true distribution.
        - compute_elbo: Computes the Negative Evidence Lower Bound (NELBO) for the VAE.
        - train: Training the VAE.
    
    """

    def __init__(self, input_dim, hidden_dims, output_dim, g_dim, A_dim):
        super(FoGSMVAE, self).__init__()
        self.encoder = VariationalEncoder(input_dim, hidden_dims, g_dim, A_dim)
        self.decoder = VariationalDecoder(output_dim, hidden_dims, g_dim, A_dim)

    @profile
    def build_lower_triangular(self, chol_flat):
        """
        Converts the flattened Cholesky parameter into a lower triangular matrix.

        """
        g_dim = chol_flat.size(1)
        num_g_points = chol_flat.size(0)
        # Create a zero matrix to store the lower triangular matrices
        L_matrices = []

        #add a live tqdm bar
        for i in tqdm(range(num_g_points)): 
            L = torch.zeros((g_dim, g_dim))
            tril_indices = torch.tril_indices(row=g_dim, col=g_dim)
            L[tril_indices[0], tril_indices[1]] = chol_flat[i]
            L_matrices.append(L)
            tqdm.write(f'Building lower triangular matrix {i+1}/{num_g_points}')
            tqdm.write(f'Cholesky parameter: {chol_flat[i]}')

        self.L_matrices = L_matrices

    def reparameterise(self,mean, chol_flat = False, log_var=None, num_samples=100):
        """
        Samples from the distribution defined by mean and logvar multiple times.

        """

        if chol_flat is False:
            # Diagonal covariance matrix case
            std = torch.exp(0.5 * log_var)
            eps = torch.randn((num_samples,) + mean.shape)
            samples = mean.unsqueeze(0) + eps * std.unsqueeze(0)
            return samples
        
        else:
            # Full covariance matrix case
            g_dim = mean.size(1)  
            num_g_points = mean.size(0) 
            samples = torch.zeros((num_g_points, num_samples, g_dim))
        
            for i in range(num_g_points):
                mean_g = mean[i]
                L = self.L_matrices[i]
            
                # Sample for the i-th GP
                eps = torch.randn((num_samples, g_dim))
                samples[i] = mean_g.unsqueeze(0) + torch.matmul(eps, L.t())

            samples = samples.permute(1, 0, 2)
            return samples
        
    def forward(self, I):
        g_mean, g_chol_flat, A_mean, A_logvar = self.encoder(I)

        print("Building lower triangular matrix for Cholesky parameter")
        self.build_lower_triangular(g_chol_flat)

        g_samples = self.reparameterise(g_mean, chol_flat=True,num_samples=1000)
        A_samples = self.reparameterise(A_mean, log_var=A_logvar,num_samples=1000)

        # Monte Carlo averaging over the samples
        g_mc_avg = torch.mean(g_samples, dim=0)  # [batch_size, g_dim]
        A_mc_avg = torch.mean(A_samples, dim=0)  # [batch_size, A_dim]

        # Concatenate averaged g and A samples for decoding
        combined_samples_avg = torch.cat([g_mc_avg, A_mc_avg], dim=-1)  # [batch_size, g_dim + A_dim]
        
        # Decode the averaged latent representation to reconstruct I
        reconstructed_I = self.decoder(combined_samples_avg)
        
        return reconstructed_I, g_mean, g_chol_flat, A_mean, A_logvar
    
    @staticmethod 
    def compute_reconstruction_loss(original_I, reconstructed_I, sigma):
        """
        Computes the reconstruction loss part of the ELBO.
    
        """
        # Compute the squared difference term
        mse_loss = F.mse_loss(reconstructed_I, original_I, reduction='sum')
        
        adjusted_loss = mse_loss / (2 * sigma**2)
    
        # Add the normalisation constant for the Gaussian distribution
        normalisation_constant = 0.5 * torch.log(2 * torch.pi * torch.tensor(sigma)**2) * torch.numel(original_I)
    
        reconstruction_loss = adjusted_loss + normalisation_constant
        return reconstruction_loss
    
    
    def compute_kl_divergence_gp(self, g_mean):
        """
        Computes the KL divergence between the mean-field approximated posterior over g
        and the GP prior, using the precomputed full covariance matrices.
        """
        num_g_points = g_mean.size(0)  # Number of Gaussian Process points
        g_dim = g_mean.size(1)  # Dimension of the Gaussian Process
        K_g = self.K_g  # GP prior covariance matrix

        Sigma = torch.matmul(self.L_matrices, self.L_matrices.permute(0, 2, 1))

        K_inv = torch.inverse(K_g)  # Invert K_g
        kl_divergence = 0
        for i in range(num_g_points):
            Sigma_i = Sigma[i]
            g_mean_i = g_mean[i]

            # Compute the KL divergence components for each GP point
            trace_term = torch.trace(torch.mm(K_inv, Sigma_i))
            quad_term = torch.mm(torch.mm(g_mean_i.unsqueeze(0), K_inv), g_mean_i.unsqueeze(1))
        
            log_det_Sigma_i = torch.logdet(Sigma_i)
            log_det_K = torch.logdet(K_g)  # log(det(K))

            kl_divergence_i = 0.5 * (trace_term + quad_term - g_dim + log_det_K - log_det_Sigma_i)
            kl_divergence += kl_divergence_i

        # Average the KL divergence over all GP points
        #kl_divergence = kl_divergence / num_g_points

        return kl_divergence


    def compute_kl_divergence(self,mean, log_var=None, chol_flat=False):
        """
        Computes the KL divergence between the approximate and the true distribution.

        """

        if chol_flat is True:
            kl_divergence = self.compute_kl_divergence_gp(mean)
        else:
            kl_divergence = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
            
        return kl_divergence
    
    def compute_elbo(self, original_I, reconstructed_I, g_mean, g_logvar, A_mean, A_logvar):
        """
        Computes the Negative Evidence Lower Bound (NELBO) for the VAE.

        """
        reconstruction_loss = self.compute_reconstruction_loss(original_I, reconstructed_I, self.sigma)
        kl_divergence_g = self.compute_kl_divergence(g_mean, chol_flat=True)
        kl_divergence_A = self.compute_kl_divergence(A_mean, log_var=A_logvar)
    
        nelbo = -reconstruction_loss - kl_divergence_g - kl_divergence_A
        return nelbo

    def train(self, I, optimiser,num_epochs=1000, print_every=100):
        """
        Training the VAE.

        """
        torch.autograd.set_detect_anomaly(True)

        elbo_values = []
        for epoch in tqdm(range(num_epochs), desc='Training', unit='epoch'):
            # Forward pass
            reconstructed_I, g_mean, g_logvar, A_mean, A_logvar = self.forward(I)
    
            # Compute the ELBO
            elbo = self.compute_elbo(I, reconstructed_I, g_mean, g_logvar, A_mean, A_logvar)
            elbo_values.append(elbo.item())
    
            # Backward pass
            optimiser.zero_grad()
            elbo.backward(retain_graph=True)
            optimiser.step()

            # Print the ELBO
            if (epoch + 1) % print_every == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], ELBO: {elbo.item()}')
                plt.figure(figsize=(9,3))
                plt.subplot(1, 3, 1)
                plt.imshow(I.detach().numpy().reshape(I.shape[1],I.shape[1]), cmap='gray')
                plt.title('Original')
                plt.axis("off")

                plt.subplot(1, 3, 2)
                plt.imshow(reconstructed_I.detach().numpy().reshape(I.shape[1],I.shape[1]), cmap='gray')
                plt.title('Reconstructed')
                plt.axis("off")

                plt.subplot(1, 3, 3) 
                plt.imshow((I - reconstructed_I).detach().numpy().reshape(I.shape[1],I.shape[1]), cmap='gray')
                plt.title('Difference')
                plt.axis("off")
                plt.show()

        return elbo_values

    

