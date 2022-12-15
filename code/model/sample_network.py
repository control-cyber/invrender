import torch.nn as nn
import torch

class SampleNetwork(nn.Module):
    '''
    Represent the intersection (sample) point as differentiable function of the implicit geometry and camera parameters.
    '''

    def forward(self, surface_output, surface_sdf_values, surface_points_grad, surface_dists, surface_cam_loc, surface_ray_dirs):
        # t -> t(theta)
        surface_ray_dirs_0 = surface_ray_dirs.detach()
        surface_points_dot = torch.bmm(surface_points_grad.view(-1, 1, 3),
                                       surface_ray_dirs_0.view(-1, 3, 1)).squeeze(-1)
        surface_dists_theta = surface_dists - (surface_output - surface_sdf_values) / surface_points_dot

        # t(theta) -> x(theta,c,v)

        #surface_points_theta_c_v = surface_cam_loc - surface_dists_theta * surface_ray_dirs;

        T=surface_dists_theta * surface_ray_dirs;

        k=np.reshape(T,[1,-1]);
        linear1=nn.Linear(np.size(k),np.size(k));
        linear2=nn.Linear(np.size(k),np.size(k),bias=False);
        sigmoid=nn.Sigmoid();
        k=linear1(k);
        k=sigmoid(k);
        k=linear1(k);
        k=linear2(k);
        surface_points_theta_c_v=np.reshape(k,np.shape(T));
        
        surface_points_theta_c_v = surface_cam_loc - T;
        

        return surface_points_theta_c_v
