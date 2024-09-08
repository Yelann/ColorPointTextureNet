<<<<<<< Updated upstream
import torch

def linear_noise_schedule(start, end, steps):
    return torch.linspace(start, end, steps)

def forward_diffusion(x0, t, noise_schedule):
    """
    x0: 原始数据
    t: 时间步
    noise_schedule: 噪声调度函数
    """
    noise = torch.randn_like(x0)
    beta_t = noise_schedule[t]
    alpha_t = 1.0 - beta_t
    xt = torch.sqrt(alpha_t) * x0 + torch.sqrt(beta_t) * noise
    return xt, noise

# 示例噪声调度
steps = 1000
beta_start = 0.0001
beta_end = 0.02
=======
import torch

def linear_noise_schedule(start, end, steps):
    return torch.linspace(start, end, steps)

def forward_diffusion(x0, t, noise_schedule):
    """
    x0: 原始数据
    t: 时间步
    noise_schedule: 噪声调度函数
    """
    noise = torch.randn_like(x0)
    beta_t = noise_schedule[t]
    alpha_t = 1.0 - beta_t
    xt = torch.sqrt(alpha_t) * x0 + torch.sqrt(beta_t) * noise
    return xt, noise

# 示例噪声调度
steps = 1000
beta_start = 0.0001
beta_end = 0.02
>>>>>>> Stashed changes
noise_schedule = linear_noise_schedule(beta_start, beta_end, steps)