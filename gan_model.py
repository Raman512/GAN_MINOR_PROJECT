import torch
import torch.nn as nn


def initialize_weight(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


class Genrator(nn.Module):
    def __init__(self, latent_vector, img_feature, num_classes, img_channel=3, kenl_size=4, st=2, padd=1):
        super(Genrator, self).__init__()
        self.embedding_layer = nn.Linear(num_classes,num_classes)
        self.linear_layer = nn.Linear(num_classes+latent_vector,latent_vector+num_classes)
        self.gen_layer = nn.Sequential(
            self.conv_tarns_layer(in_channel=latent_vector+num_classes, out_channel=img_feature * 2, kernel_size=kenl_size,
                                  stride=1, padding=0),
            self.conv_tarns_layer(in_channel=img_feature * 2, out_channel=img_feature, kernel_size=kenl_size, stride=st,
                                  padding=padd),
            self.conv_tarns_layer(in_channel=img_feature, out_channel=int(img_feature * (1 / 2)), kernel_size=kenl_size,
                                  stride=st, padding=padd),
            self.conv_tarns_layer(in_channel=int(img_feature * (1 / 2)), out_channel=int(img_feature * (1 / 4)),
                                  kernel_size=kenl_size, stride=st, padding=padd),
            self.conv_tarns_layer(in_channel=int(img_feature * (1 / 4)), out_channel=int(img_feature * (1 / 8)),
                                  kernel_size=kenl_size, stride=st, padding=padd),
            self.conv_tarns_layer(in_channel=int(img_feature * (1 / 8)), out_channel=int(img_feature * (1 / 16)),
                                  kernel_size=kenl_size, stride=st, padding=padd),
            nn.ConvTranspose2d(in_channels=int(img_feature * (1 / 16)), out_channels=img_channel, kernel_size=kenl_size,
                               stride=st, padding=padd),
            nn.Tanh(),
        )

    def forward(self, x, labels):
        labels = self.embedding_layer(labels)
        x = self.linear_layer(torch.cat([x.view(x.shape[0],-1), labels], dim=1))
        return self.gen_layer(x.unsqueeze(2).unsqueeze(3))

    def conv_tarns_layer(self, in_channel, out_channel, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel,kernel_size=3,stride=1,padding=1),
            nn.ConvTranspose2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride,
                               padding=padding),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
        )


class Discriminator(nn.Module):
    def __init__(self, img_channel, img_feature, num_classes, image_size=256, knl_size=4, st=2, padd=1):
        super(Discriminator, self).__init__()
        self.img_size = image_size
        self.embedding_layer = nn.Linear(num_classes, image_size*image_size)
        self.disc_layers = nn.Sequential(
            nn.Conv2d(img_channel+1, int(img_feature * (1 / 32)), kernel_size=knl_size, stride=st, padding=padd),
            nn.LeakyReLU(0.2),
            self.conv_layer(int(img_feature * (1 / 32)), int(img_feature * (1 / 16)), kernel_size=knl_size, stride=st,
                            padding=padd),
            self.conv_layer(int(img_feature * (1 / 16)), int(img_feature * (1 / 8)), kernel_size=knl_size, stride=st,
                            padding=padd),
            self.conv_layer(int(img_feature * (1 / 8)), int(img_feature * (1 / 4)), kernel_size=knl_size, stride=st,
                            padding=padd),
            self.conv_layer(int(img_feature * (1 / 4)), int(img_feature * (1 / 2)), kernel_size=knl_size, stride=st,
                            padding=padd),
            self.conv_layer(int(img_feature * (1 / 2)), img_feature, kernel_size=knl_size, stride=st, padding=padd),
            nn.Conv2d(img_feature, 1, kernel_size=knl_size, stride=st, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x, labels):
        embedding_channel = self.embedding_layer(labels.float())
        x = torch.cat([x, embedding_channel.view(labels.shape[0], 1, self.img_size, self.img_size)], dim=1)
        return self.disc_layers(x)

    def conv_layer(self, in_channel, out_channel, kernel_size, stride=1, padding=1):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel,kernel_size=3,stride=1,padding=1),
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, padding=padding,
                      stride=stride, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.2),
        )
