import torch
import torch.nn as nn


class Trainer:
    def __init__(self, encoder: nn.Module, decoder: nn.Module, data_discriminator: nn.Module,
                 latent_discriminator: nn.Module, batch_size, window_size):
        self.encoder = encoder
        self.decoder = decoder
        self.data_discriminator = data_discriminator
        self.latent_discriminator = latent_discriminator

        self.adversarial_criterion = torch.nn.BCEWithLogitsLoss()
        self.reconstruction_criterion = torch.nn.L1Loss(reduction='none')

        self.input_noise = lambda: (torch.randn(batch_size, window_size) * 0.01).cuda()
        self.label_real = torch.ones(batch_size, 1).cuda()
        self.label_fake = torch.zeros(batch_size, 1).cuda()

    def print_model(self):
        print(self.encoder)
        print(self.decoder)
        print(self.data_discriminator)
        print(self.latent_discriminator)

    def train(self):
        self.encoder.train()
        self.decoder.train()
        self.data_discriminator.train()
        self.latent_discriminator.train()

    def eval(self):
        self.encoder.eval()
        self.decoder.eval()
        self.data_discriminator.eval()
        self.latent_discriminator.eval()

    def data_gen_loss(self, x, y=None, margin=1.0):
        x_rec = self.decoder(self.encoder(x + self.input_noise()))
        data_gen_loss = self.adversarial_criterion(self.data_discriminator(x_rec), self.label_real)

        if y is None:
            rec_loss = self.adversarial_criterion(x_rec, x).mean()
        else:
            rec = self.reconstruction_criterion(x_rec, x)
            rec_loss = ((1 - y) * rec + y * torch.max(torch.zeros_like(rec), margin - rec)).mean()

        return data_gen_loss, rec_loss

    def data_dis_loss(self, x):
        x_rec = self.decoder(self.encoder(x + self.input_noise()))
        data_dis_loss = self.adversarial_criterion(self.data_discriminator(x_rec), self.label_fake) + \
                        self.adversarial_criterion(self.data_discriminator(x), self.label_real)

        return data_dis_loss

    def latent_gen_loss(self, x):
        z_hat = self.encoder(x + self.input_noise())
        latent_gen_loss = self.adversarial_criterion(self.latent_discriminator(z_hat), self.label_real)

        return latent_gen_loss

    def latent_dis_loss(self, x):
        z_hat = self.encoder(x + self.input_noise())
        z_prior = torch.randn_like(z_hat).cuda()
        latent_dis_loss = self.adversarial_criterion(self.latent_discriminator(z_hat),
                                                self.label_fake) + self.adversarial_criterion(
            self.latent_discriminator(z_prior), self.label_real)

        return latent_dis_loss

    def reconstruct(self, x):
        out = self.decoder(self.encoder(x))

        return out
