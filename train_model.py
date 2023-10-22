import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import utils
from tqdm import tqdm
import torchvision.transforms as transforms
#from PIL import Image
#import datetime
from gan_model import Genrator,Discriminator
#from tensorboard import h

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
learning_rate = 0.00002
img_size = 256
Batch_size = 9
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]
)

dataset = utils.CelebADataset(transform=transform)
data = DataLoader(dataset, batch_size=Batch_size)

def train_model(genr, discrim, opt_gen, opt_disc, loader, z_dim, LAMBDA_GP=3, device=torch.device('cuda')):
    genr.train()
    discrim.train()
    for  real,labels in tqdm(loader):
        real = real.to(device)
        batch_size= real.shape[0]
        labels = labels.float().to(device)
        for _ in range(2):
            noise = torch.randn(batch_size, z_dim, 1, 1).to(device)
            fake = genr(noise, labels).to(device)
            discrim_real = discrim(real, labels).reshape(-1)
            discrim_fake = discrim(fake, labels).reshape(-1)
            gp = utils.gradient_penalty(discrim, real ,labels, fake, device=device)
            loss_discrim = (
                -(torch.mean(discrim_real) - torch.mean(discrim_fake)) + LAMBDA_GP * gp
            )
            discrim.zero_grad()
            loss_discrim.backward(retain_graph=True)
            opt_disc.step()
        # Train Generator: max E[discrim(gen_fake)] <-> min -E[discrim(gen_fake)]
        gen_fake = discrim(fake, labels).reshape(-1)
        loss_gen = -torch.mean(gen_fake)
        genr.zero_grad()
        loss_gen.backward()
        opt_gen.step()
    # _,labels = loader
    # noise = torch.randn(Batch_size, z_dim, 1, 1).to(device)
    # fake = genr(noise, labels).to('cpu')    
    # fake = torchvision.utils.make_grid(fake, normalize=True).to('cpu')   
    # fake_image = r'D:\minor_project_gan\datasets\celeb_a\fake_img'
    # img = Image.fromarray((fake.numpy() * 255).astype('uint8').transpose((1, 2, 0)))
    # img.save(rf'{fake_image}\{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.jpg', 'JPEG')
        
        
z = 0
z_dim = 100    
discriminator_path = r"D:\minor_project_gan\datasets\celeb_a\discriminator"
genrator_path = r"D:\minor_project_gan\datasets\celeb_a\genrator"
genr = Genrator(z_dim, 1024, 8).to(torch.device('cuda'))
genr.train()
discrim = Discriminator(3, 1024, 8).to(torch.device('cuda'))
discrim.train()
genr.load_state_dict(torch.load(rf"{genrator_path}\genrator{z}.pt"))
discrim.load_state_dict(torch.load(rf"{discriminator_path}\discriminator{z}.pt"))
genr_optim = optim.Adam(genr.parameters(), lr=learning_rate)
discrim_optim = optim.Adam(discrim.parameters(), lr=learning_rate)

i = 1

for k in range(10000):

    train_model(genr, discrim, genr_optim, discrim_optim, data, z_dim, device=torch.device('cuda'))
    torch.save(genr.state_dict(), f'D:\minor_project_gan\datasets\celeb_a\genrator\genrator{i}.pt')
    torch.save(discrim.state_dict(), f'D:\minor_project_gan\datasets\celeb_a\discriminator\discriminator{i}.pt')
    print(i)
    i = utils.buffer_value(i, 5)
    print(k)