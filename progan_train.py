import torch
from torch import optim
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from models import Discriminator, Generator
from math import log2

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
torch.backends.cudnn.benchmarks = True

def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location="cuda")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def grad_pen(critic, real, fake, alpha, train_step, device="cuda"):
    batch_size, c, h, w = real.shape
    eps = torch.rand((batch_size, 1, 1, 1)).repeat(1, c, h, w).to(device)
    inter_img = real * eps + fake * (1 - eps)
    inter_img.requires_grad_(True)

    mix_score = critic(inter_img, alpha, train_step)

    grad = torch.autograd.grad(
        inputs=inter_img,
        outputs=mix_score,
        grad_outputs=torch.ones_like(mix_score),
        create_graph=True,
        retain_graph=True,
    )[0]
    grad = grad.view(grad.shape[0], -1)
    grad_norm = grad.norm(2, dim=1)
    return torch.mean((grad_norm - 1) ** 2)

in_transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
train_img_size = 128
data_path = "cats"+str(train_img_size)
data = torchvision.datasets.ImageFolder(root=data_path,transform=in_transform) #,train=True
save_model = True
load_model = True
learning_rate = 1e-3
batch_sizes = [16, 16, 16, 16, 16, 8, 8, 8, 4]
step_of_scaling=int(log2(train_img_size)-2)
loader = DataLoader(data, batch_size=batch_sizes[step_of_scaling], shuffle=True)
img_channels = 3
z_dim = 256
in_channels = 256
lambda_gp= 10
fixed_noise = torch.randn(8, z_dim, 1, 1).to(device)
num_epochs=1000
gen = Generator(z_dim,in_channels, img_channels=img_channels).to(device)
disc = Discriminator(in_channels, img_channels=img_channels).to(device)
opt_gen = optim.Adam(gen.parameters(), lr=learning_rate, betas=(0.0, 0.99))
opt_disc = optim.Adam(disc.parameters(), lr=learning_rate, betas=(0.0, 0.99))
scaler_disc = torch.cuda.amp.GradScaler()
scaler_gen = torch.cuda.amp.GradScaler()
writer_real = SummaryWriter(f"logs/real2")
writer_fake = SummaryWriter(f"logs/fake2")
alpha=0
board_step=0
with open("board_step.txt","r") as board_step_r: board_step=int(board_step_r.readline())
if load_model:
    load_checkpoint("gen.pth", gen, opt_gen, learning_rate,)
    load_checkpoint("disc.pth", disc, opt_disc, learning_rate,)

gen.train()
disc.train()

for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.to(device)
        cur_batch_size = real.shape[0]

        noise = torch.randn(cur_batch_size, z_dim, 1, 1).to(device)

        with torch.cuda.amp.autocast():
            fake = gen(noise, alpha, step_of_scaling)
            disc_real = disc(real, alpha, step_of_scaling)
            disc_fake = disc(fake.detach(), alpha, step_of_scaling)
            gp = grad_pen(disc, real, fake, alpha, step_of_scaling, device=device)
            loss_disc = (
                    -(torch.mean(disc_real) - torch.mean(disc_fake))
                    + lambda_gp * gp
                    + (0.001 * torch.mean(disc_real ** 2))
            )

        opt_disc.zero_grad()
        scaler_disc.scale(loss_disc).backward(retain_graph=True)
        scaler_disc.step(opt_disc)
        scaler_disc.update()

        with torch.cuda.amp.autocast():
            gen_fake = disc(fake, alpha, step_of_scaling)
            loss_gen = -torch.mean(gen_fake)

        opt_gen.zero_grad()
        scaler_gen.scale(loss_gen).backward()
        scaler_gen.step(opt_gen)
        scaler_gen.update()

        alpha += cur_batch_size / 81660
        alpha = min(alpha, 1)

        if batch_idx % 100 == 0:
            with torch.no_grad():
                fake = gen(fixed_noise, alpha, step_of_scaling)
                img_grid_real = torchvision.utils.make_grid(real[:8], normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake[:8], normalize=True)

                writer_real.add_image("Real2", img_grid_real, global_step=board_step)
                writer_fake.add_image("Fake2", img_grid_fake, global_step=board_step)
            if save_model:
                save_checkpoint(gen, opt_gen, filename="gen.pth")
                save_checkpoint(disc, opt_disc, filename="disc.pth")
            board_step+=1
            with open("board_step.txt","w") as board_step_w: board_step_w.write(str(board_step))

            print(epoch,board_step,alpha)
            print(f"D_loss:{loss_disc}\nG_loss:{loss_gen}")
