from ._imports import *
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
    
    def log_likelihood(self, I, g,A):
        A = A.repeat(len(self.thetas))
        I_hat = g * A
        I_hat = torch.sum(I_hat.reshape(len(self.thetas), self.grid_size, self.grid_size), dim=0)
        return MultivariateNormal(I_hat.flatten(), self.sigma * torch.eye(self.grid_size**2)).log_prob(I.flatten())

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

    def generate_fogsm_dataset(self, num_samples, save=False,save_path=None):
        samples = []
        for _ in range(num_samples):
            I, g = self.samples()
            samples.append((I, g))
    
        # Convert samples to tensors
        images, gs = zip(*samples)
        images = torch.stack(images)
        gs = torch.stack(gs)
    
        if save:
            torch.save(images, gs, save_path)
        else:
            return images

    def visualise_samples(self, save_path, num_samples_to_visualise, grid_size):
        # Load the saved samples
        images, _ = torch.load(save_path)

        # Select a subset of samples to visualise
        selected_samples = images[:num_samples_to_visualise]

        # Create a grid of images
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
        fig.subplots_adjust(hspace=0.1, wspace=0.1)

        for i, ax in enumerate(axes.flat):
            if i < num_samples_to_visualise:
                image = selected_samples[i].detach().numpy()
                image = (image - image.min()) / (image.max() - image.min())

                ax.imshow(image, cmap='gray')
        
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis('off')

        plt.tight_layout()
        plt.show()

    

