import torch
import utils
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import datetime
# # from torchvision.datasets import MNIST
# # height = 28
# # width = 28
# # torch.backends.cudnn.enabled = False
# #
# # transform = transforms.Compose([
# #                 transforms.Resize((height, width)),
# #                 transforms.ToTensor(),
# #                 transforms.Normalize((0.13,),(0.2,))
# #      ])
# #
# # train_dataset = MNIST(root="./data", transform=transform, download=True, train=True)
# # test_dataset = MNIST(root="./data", transform=transform, download=True, train=False)
# # train_data_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, pin_memory=True)
# #
# # for batch in train_data_loader:
# #     print(batch)
# #     break
# import matplotlib.pyplot as plt
#
# # create data
# x = [1, 2, 3, 4, 5]
# y = [2, 4, 6, 8, 10]
#
# # create plot
# plt.plot(x, y)
#
# # add labels and title
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.title('Line Graph Example')
#
# # display plot
# plt.show()


from gan_model import Genrator, Discriminator, initialize_weight
import torch
# for i in range(5):
#     my_model = Genrator(100, 1024, 8)
#     my_model2 = Discriminator(3, 1024, 8)
#     initialize_weight(my_model)
#     initialize_weight(my_model2)
#     torch.save(my_model.state_dict(), f'D:\minor_project_gan\datasets\celeb_a\genrator\genrator{i}.pt')
#     torch.save(my_model2.state_dict(), f'D:\minor_project_gan\datasets\celeb_a\discriminator\discriminator{i}.pt')
#     print("done")

# def forward(self, x, labels):
#         embed = self.embedding_layer(labels).unsqueeze(2).unsqueeze(3)
#         embed = embed.to(x.device)
#         x = torch.cat([x, embed], dim=1).to(embed.device)
#         return self.gen_layer(x)



genrator_path = r"D:\coding\datasets\celeb_a\genrator"

genr = Genrator(100, 1024, 8)

genr.load_state_dict(torch.load(rf"{genrator_path}\genrator0.pt"))
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]
)

dataset = utils.CelebADataset(transform=transform)
data = DataLoader(dataset, batch_size=7)
for real,labels in data:
    noise = torch.randn(7, 100, 1, 1)
    fake = genr(noise, labels.float())  
    fake = torchvision.utils.make_grid(fake, normalize=True).to('cpu')   
    fake_image = r'D:\minor_project_gan\datasets\celeb_a\fake_img'
    img = Image.fromarray((fake.numpy() * 255).astype('uint8').transpose((1, 2, 0)))
    real = torchvision.utils.make_grid(real,normalize=True)
    img2 = Image.fromarray((real.numpy() * 255).astype('uint8').transpose((1, 2, 0)))
    img2.show()
    #img.show()
    img.save(rf'{fake_image}\{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.jpg', 'JPEG')
    break