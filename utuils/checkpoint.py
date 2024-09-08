import torch

def save_checkpoint(model, vae, sampler, optimizer, optimizer_sam, save_path, save_samplenet=False):
    if save_samplenet:
        torch.save({
            'model': model.state_dict(), 
            'vae': vae.state_dict(),
            'samplenet': sampler.state_dict(),
            'optimizer': optimizer.state_dict(),
            'optimizer_sam': optimizer_sam.state_dict(),
        }, save_path)
    else:
        torch.save({
            'model': model.state_dict(), 
            'vae': vae.state_dict(),
            'optimizer': optimizer.state_dict()
        }, save_path)

def save_vae(vae, optimizer, save_path):
    torch.save({
        'vae': vae.state_dict(),
        'optimizer': optimizer.state_dict()
    }, save_path)

def load_checkpoint(model, vae, sampler, optimizer, optimizer_sam, load_path, load_samplenet=False, load_optimizer=True):
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model'])
    vae.load_state_dict(checkpoint['vae'])
    if load_optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if load_samplenet:
        sampler.load_state_dict(checkpoint['samplenet'])
        optimizer_sam.load_state_dict(checkpoint['optimizer_sam'])


def load_model(model, vae, load_path):
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model'])
    vae.load_state_dict(checkpoint['vae'])


def load_vae(vae, load_path):
    checkpoint = torch.load(load_path)
    vae.load_state_dict(checkpoint['vae'])
    # model.load_state_dict(checkpoint['model'])

def load_samplenet(sampler, load_path):
    checkpoint = torch.load(load_path)
    sampler.load_state_dict(checkpoint['samplenet'])

def load_samplenet2(sampler, load_path):
    checkpoint = torch.load(load_path)
    sampler.load_state_dict(checkpoint)