import torch

def save_checkpoint(model, optimizer, save_path):
    torch.save({
        'model': model.state_dict(), 
        'vae': model.vae.state_dict(),
        'optimizer': optimizer.state_dict()
    }, save_path)


def load_checkpoint(model, optimizer, load_path):
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model'])
    model.vae.load_state_dict(checkpoint['vae'])
    optimizer.load_state_dict(checkpoint['optimizer'])


def load_model(model, load_path):
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model'])
    model.vae.load_state_dict(checkpoint['vae'])


def load_vae(model, load_path):
    checkpoint = torch.load(load_path)
    model.vae.load_state_dict(checkpoint['vae'])