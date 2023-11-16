import os
import argparse
import numpy
import torch
from torch.autograd import Variable
import torchvision.utils as vutils
import random
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import pathlib
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from prettytable import PrettyTable
from lib import datasets, models, utils


def main(args):
    device = utils.select_device(args.cuda_device)

    # weights_path = 'Output/GANlosses/'

    # if not os.path.exists(weights_path):
    #     path = pathlib.Path(weights_path)
    #     path.mkdir(parents=True)

    # Setting the title for the file saved
    # if args.loss_D == 'DCGAN':
    #     title = 'GAN_'
    # if args.loss_D == 'LSGAN':
    #     title = 'LSGAN_'
    # if args.loss_D == 'WGAN-GP':
    #     title = 'WGANGP_'
    # if args.loss_D == 'HingeGAN':
    #     title = 'HingeGAN_'
    # if args.loss_D == 'RSGAN':
    #     title = 'RSGAN_'
    # if args.loss_D == 'RaSGAN':
    #     title = 'RaSGAN_'
    # if args.loss_D == 'RaLSGAN':
    #     title = 'RaLSGAN_'
    # if args.loss_D == 'RaHingeGAN':
    #     title = 'RaHingeGAN_'
    #
    # if args.seed is not None:
    #     title = title + 'seed%i' % args.seed

    # Check folder run-i for all i=0,1,... until it finds run-j which does not exists, then creates a new folder run-j
    # run = 0
    # base_dir = f"{args.output_folder}/{title}-{run}"
    # while os.path.exists(base_dir):
    #     run += 1
    #     base_dir = f"{args.output_folder}/{title}-{run}"
    # os.mkdir(base_dir)
    # logs_dir = f"{base_dir}/logs"
    # os.mkdir(logs_dir)
    # os.mkdir(f"{base_dir}/images")
    # if args.gen_extra_images > 0 and not os.path.exists(f"{args.extra_folder}"):
    #     os.mkdir(f"{args.extra_folder}")
    #
    # # where we save the output
    # log_output = open(f"{logs_dir}/log.txt", 'w')
    # print(args)
    # print(args, file=log_output)

    # For plotting the Loss of discriminator and generator using tensorboard
    # To fix later, not compatible with using tensorflow
    # from tensorboard_logger import configure, log_value
    # configure(logs_dir, flush_secs=5)

    if args.cuda:
        cudnn.deterministic = True
        cudnn.benchmark = True

    torch.utils.backcompat.broadcast_warning.enabled = True

    # Setting seed

    if args.seed is None:
        args.seed = random.randint(1, 10000)
    # print(f"Random Seed: {args.seed}")
    # print(f"Random Seed: {args.seed}", file=log_output)
    random.seed(args.seed)
    numpy.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)

    dataset = datasets.make_dataset(dataset=args.dataset,
                                    batch_size=args.batch_size,
                                    img_size=args.image_size,
                                    classification=False,
                                    artificial=False,
                                    train=True)

    # Loading data randomly
    # def generate_random_sample():
    #     while True:
    #         random_indexes = numpy.random.choice(data.__len__(), size=args.batch_size, replace=False)
    #         batch = [data[i][0] for i in random_indexes]
    #         yield torch.stack(batch, 0)

    # random_sample = generate_random_sample()

    # Load models
    generator = models.Generator(args).apply(models.weights_init).to(device)
    discriminator = models.Discriminator(args).apply(models.weights_init).to(device)

    # Criterion
    criterion = torch.nn.BCELoss().to(device)
    bce_stable = torch.nn.BCEWithLogitsLoss().to(device)
    bce_stable_no_reduce = torch.nn.BCEWithLogitsLoss(reduction='none').to(device)

    real_label = 1.
    fake_label = 0.

    # Soon to be variables
    # x = torch.FloatTensor(args.batch_size, args.n_colors, args.image_size, args.image_size)
    # x_fake = torch.FloatTensor(args.batch_size, args.n_colors, args.image_size, args.image_size)
    # Weighted sum of fake and real image, for gradient penalty
    x_both = torch.FloatTensor(args.batch_size, args.n_colors, args.image_size, args.image_size).to(device)
    # Uniform weight
    u = torch.FloatTensor(args.batch_size, 1, 1, 1).to(device)
    # This is to see during training, size and values won't change
    z_test = torch.FloatTensor(args.batch_size, args.z_size, 1, 1).normal_(0, 1).to(device)
    # For the gradients, we need to specify which one we want and want them all
    grad_outputs = torch.ones(args.batch_size).to(device)

    # Everything cuda
    # if args.cuda:
    #     criterion = criterion.cuda()
    #     bce_stable.cuda()
    #     bce_stable_no_reduce.cuda()
    #     # x = x.cuda()
    #     # x_fake = x_fake.cuda()
    #     x_both = x_both.cuda()
    #     u = u.cuda()
    #     z_test = z_test.cuda()
    #     grad_outputs = grad_outputs.cuda()

    # Now Variables
    # x = Variable(x)
    # x_fake = Variable(x_fake)
    z_test = Variable(z_test)

    # Based on DCGAN paper, they found using betas[0]=.50 better.
    # betas[0] represent is the weight given to the previous mean of the gradient
    # betas[1] is the weight given to the previous variance of the gradient
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.lr_D, betas=(args.beta1, args.beta2),
                                               weight_decay=args.weight_decay)
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr_G, betas=(args.beta1, args.beta2),
                                           weight_decay=args.weight_decay)

    # exponential weight decay on lr
    discriminator_decay = torch.optim.lr_scheduler.ExponentialLR(discriminator_optimizer, gamma=1 - args.decay)
    generator_decay = torch.optim.lr_scheduler.ExponentialLR(generator_optimizer, gamma=1 - args.decay)

    # Load existing models
    if args.load:
        checkpoint = torch.load(args.load)
        current_set_images = checkpoint['current_set_images']
        iter_offset = checkpoint['i']
        generator.load_state_dict(checkpoint['G_state'])
        discriminator.load_state_dict(checkpoint['D_state'])
        generator_optimizer.load_state_dict(checkpoint['G_optimizer'])
        discriminator_optimizer.load_state_dict(checkpoint['D_optimizer'])
        generator_decay.load_state_dict(checkpoint['G_scheduler'])
        discriminator_decay.load_state_dict(checkpoint['D_scheduler'])
        z_test.copy_(checkpoint['z_test'])
        del checkpoint
        print(f'Resumed from iteration {current_set_images * args.gen_every}.')
    else:
        current_set_images = 0
        iter_offset = 0

    # print(generator)
    # print(generator, file=log_output)
    # print(discriminator)
    # print(discriminator, file=log_output)

    fixed_noise = torch.randn(32, args.z_size, 1, 1, device=device)

    writer = SummaryWriter()
    print("Starting Training Loop...")
    for epoch in range(iter_offset, args.epochs):
        # Fake images saved
        running_loss_g = 0.0
        running_loss_d = 0.0
        # fake_test = generator(z_test)
        # vutils.save_image(fake_test.data, '%s/images/fake_samples_iter%05d.png' % (base_dir, epoch), normalize=True)

        with tqdm(total=len(dataset), desc="Epoch {}".format(epoch + 1)) as pbar:
            for data in dataset:
                real = data[0].to(device)
                current_batch_size = real.size(0)

                for p in discriminator.parameters():
                    p.requires_grad = True

                for _ in range(args.Diters):

                    ##########################
                    # (1) Update discriminator
                    ##########################

                    discriminator.zero_grad()
                    # images = random_sample.__next__()
                    # Mostly necessary for the last one because if N might not be a multiple of batch_size
                    # current_batch_size = images.size(0)
                    # if args.cuda:
                    #     images = images.cuda()
                    # Transfer batch of images to x
                    # x.data.resize_as_(images).copy_(images)
                    # del images

                    y_pred = discriminator(real)

                    if args.loss_D in ['DCGAN', 'LSGAN', 'WGAN-GP', 'HingeGAN']:
                        # Train with real data
                        y = torch.full((current_batch_size,), real_label, dtype=torch.float, device=device) # y.data.resize_(current_batch_size).fill_(1)
                        if args.loss_D == 'DCGAN':
                            errD_real = criterion(y_pred, y)
                        if args.loss_D == 'LSGAN':
                            errD_real = torch.mean((y_pred - y) ** 2)
                        if args.loss_D == 'WGAN-GP':
                            errD_real = -torch.mean(y_pred)
                        if args.loss_D == 'HingeGAN':
                            errD_real = torch.mean(torch.nn.ReLU()(1.0 - y_pred))

                        errD_real.backward()

                        # Train with fake data
                        z = torch.randn(current_batch_size, args.z_size, 1, 1, device=device)
                        fake = generator(z)
                        # x_fake.data.resize_(fake.data.size()).copy_(fake.data)
                        y = torch.full((current_batch_size,), fake_label, dtype=torch.float,
                                       device=device)  # y.data.resize_(current_batch_size).fill_(0)
                        # Detach y_pred from the neural network generator and put it inside discriminator
                        y_pred_fake = discriminator(fake.detach())

                        if args.loss_D == 'DCGAN':
                            errD_fake = criterion(y_pred_fake, y)
                        if args.loss_D == 'LSGAN':
                            errD_fake = torch.mean(y_pred_fake ** 2)
                        if args.loss_D == 'WGAN-GP':
                            errD_fake = torch.mean(y_pred_fake)
                        if args.loss_D == 'HingeGAN':
                            errD_fake = torch.mean(torch.nn.ReLU()(1.0 + y_pred_fake))

                        errD_fake.backward()
                        errD = errD_real + errD_fake
                    else:
                        y = torch.full((current_batch_size,), real_label, dtype=torch.float,
                                       device=device)  # y.data.resize_(current_batch_size).fill_(1)
                        y2 = torch.full((current_batch_size,), fake_label, dtype=torch.float,
                                        device=device)  # y2.data.resize_(current_batch_size).fill_(0)
                        z = torch.randn(current_batch_size, args.z_size, 1, 1, device=device)

                        fake = generator(z)
                        # x_fake.data.resize_(fake.data.size()).copy_(fake.data)
                        y_pred_fake = discriminator(fake.detach())

                        if args.loss_D == 'RSGAN':
                            errD = bce_stable(y_pred - y_pred_fake, y)
                        if args.loss_D == 'RaSGAN':
                            errD = (bce_stable(y_pred - torch.mean(y_pred_fake), y) + bce_stable(
                                y_pred_fake - torch.mean(y_pred),
                                y2)) / 2
                        if args.loss_D == 'RaLSGAN':  # (y_hat-1)^2 + (y_hat+1)^2
                            errD = (torch.mean((y_pred - torch.mean(y_pred_fake) - y) ** 2) + torch.mean(
                                (y_pred_fake - torch.mean(y_pred) + y) ** 2)) / 2
                        if args.loss_D == 'RaHingeGAN':
                            errD = (torch.mean(torch.nn.ReLU()(1.0 - (y_pred - torch.mean(y_pred_fake)))) + torch.mean(
                                torch.nn.ReLU()(1.0 + (y_pred_fake - torch.mean(y_pred))))) / 2
                        errD_real = errD
                        errD_fake = errD
                        errD.backward()

                    if args.loss_D in ['WGAN-GP'] or args.grad_penalty:
                        # Gradient penalty
                        u.data.resize_(current_batch_size, 1, 1, 1)
                        u.uniform_(0, 1)
                        x_both = real * u + fake * (1 - u)
                        if args.cuda:
                            x_both = x_both.cuda()
                        # We only want the gradients with respect to x_both
                        x_both = Variable(x_both, requires_grad=True)
                        grad = torch.autograd.grad(outputs=discriminator(x_both), inputs=x_both, grad_outputs=grad_outputs,
                                                   retain_graph=True,
                                                   create_graph=True, only_inputs=True)[0]
                        # We need to norm 3 times (over n_colors x image_size x image_size) to get only a vector of size "batch_size"
                        grad_penalty = args.penalty * ((grad.norm(2, 1).norm(2, 1).norm(2, 1) - 1) ** 2).mean()
                        grad_penalty.backward()
                    discriminator_optimizer.step()

                ########################
                # (2) Update generator network #
                ########################

                # Make it a tiny bit faster
                for p in discriminator.parameters():
                    p.requires_grad = False

                for t in range(args.Giters):

                    generator.zero_grad()
                    y = torch.full((current_batch_size,), real_label, dtype=torch.float,
                                   device=device)  # y.data.resize_(current_batch_size).fill_(1)
                    z = torch.randn(current_batch_size, args.z_size, 1, 1, device=device)
                    fake = generator(z)
                    y_pred_fake = discriminator(fake)

                    # if args.loss_D not in ['DCGAN', 'LSGAN', 'WGAN-GP', 'HingeGAN']:
                    #     images = random_sample.__next__()
                    #     current_batch_size = images.size(0)
                    #     if args.cuda:
                    #         images = images.cuda()
                    #     x.data.resize_as_(images).copy_(images)
                    #     del images

                    if args.loss_D == 'DCGAN':
                        errG = criterion(y_pred_fake, y)
                    if args.loss_D == 'LSGAN':
                        errG = torch.mean((y_pred_fake - y) ** 2)
                    if args.loss_D == 'WGAN-GP':
                        errG = -torch.mean(y_pred_fake)
                    if args.loss_D == 'HingeGAN':
                        errG = -torch.mean(y_pred_fake)
                    if args.loss_D == 'RSGAN':
                        y_pred = discriminator(real)
                        # Non-saturating
                        errG = bce_stable(y_pred_fake - y_pred, y)
                    if args.loss_D == 'RaSGAN':
                        y_pred = discriminator(real)
                        # Non-saturating
                        y2 = torch.full((current_batch_size,), fake_label, dtype=torch.float,
                                        device=device)  # y2.data.resize_(current_batch_size).fill_(0)
                        errG = (bce_stable(y_pred - torch.mean(y_pred_fake), y2) + bce_stable(y_pred_fake - torch.mean(y_pred),
                                                                                              y)) / 2
                    if args.loss_D == 'RaLSGAN':
                        y_pred = discriminator(real)
                        errG = (torch.mean((y_pred - torch.mean(y_pred_fake) + y) ** 2) + torch.mean(
                            (y_pred_fake - torch.mean(y_pred) - y) ** 2)) / 2
                    if args.loss_D == 'RaHingeGAN':
                        y_pred = discriminator(real)
                        # Non-saturating
                        errG = (torch.mean(torch.nn.ReLU()(1.0 + (y_pred - torch.mean(y_pred_fake)))) + torch.mean(
                            torch.nn.ReLU()(1.0 - (y_pred_fake - torch.mean(y_pred))))) / 2
                    errG.backward()
                    D_G = y_pred_fake.data.mean()
                    generator_optimizer.step()
                discriminator_decay.step()
                generator_decay.step()

                pbar.update(1)

            ###############
            # Saving the results
            ###############
            writer.add_scalar(args.loss_D + '/loss/generator', running_loss_g / len(dataset.dataset), epoch)
            writer.add_scalar(args.loss_D + '/loss/discriminator', running_loss_d / len(dataset.dataset), epoch)

            # Save images of the epoch
            with torch.no_grad():
                fake = generator(fixed_noise)
                img_grid_fake = make_grid(fake[:32], normalize=True)
                writer.add_image(args.loss_D.upper(), img_grid_fake, global_step=epoch)

            # Log results so we can see them in TensorBoard after
            # log_value('Diff', -(errD.data.item()+errG.data.item()), i)
            # log_value('errD', errD.data.item(), i)
            # log_value('errG', errG.data.item(), i)

        # if (epoch + 1) % args.print_every == 0:
        #     fmt = '[%d] Diff: %.4f loss_D: %.4f loss_G: %.4f time:%.4f'
        #     s = fmt % (epoch, -errD.data.item() + errG.data.item(), errD.data.item(), errG.data.item(), end - start)
        #     print(s)
        #     print(s, file=log_output)

        # Evaluation metrics
        if (epoch + 1) % args.gen_every == 0:

            current_set_images += 1

            # Save models
            if args.save:
                if not os.path.exists('%s/models/' % (args.extra_folder)):
                    os.mkdir('%s/models/' % (args.extra_folder))
                torch.save({
                    'i': epoch + 1,
                    'current_set_images': current_set_images,
                    'G_state': generator.state_dict(),
                    'D_state': discriminator.state_dict(),
                    'G_optimizer': generator_optimizer.state_dict(),
                    'D_optimizer': discriminator_optimizer.state_dict(),
                    'G_scheduler': generator_decay.state_dict(),
                    'D_scheduler': discriminator_decay.state_dict(),
                    'z_test': z_test,
                }, '%s/models/state_%02d.pth' % (args.extra_folder, current_set_images))
                s = 'Models saved'
                # print(s)
                # print(s, file=log_output)

            # Delete previously existing images
            if os.path.exists('%s/%01d/' % (args.extra_folder, current_set_images)):
                for root, dirs, files in os.walk('%s/%01d/' % (args.extra_folder, current_set_images)):
                    for f in files:
                        os.unlink(os.path.join(root, f))
            else:
                os.mkdir('%s/%01d/' % (args.extra_folder, current_set_images))

            # Generate 50k images for FID/Inception to be calculated later (not on this script, since running both tensorflow and pytorch at the same time cause issues)
            ext_curr = 0
            z_extra = torch.FloatTensor(100, args.z_size, 1, 1)
            if args.cuda:
                z_extra = z_extra.cuda()
            for ext in range(int(args.gen_extra_images / 100)):
                fake_test = generator(Variable(z_extra.normal_(0, 1)))
                for ext_i in range(100):
                    vutils.save_image((fake_test[ext_i].data * .50) + .50,
                                      '%s/%01d/fake_samples_%05d.png' % (
                                      args.extra_folder, current_set_images, ext_curr),
                                      normalize=False, padding=0)
                    ext_curr += 1
            del z_extra
            del fake_test


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--loss_D', type=str, default='DCGAN',
                        choices=['DCGAN', 'LSGAN', 'WGAN-GP', 'HingeGAN', 'RSGAN', 'RaSGAN', 'RaLSGAN', 'RaHingeGAN'])
    parser.add_argument('--dataset', '-d',
                        type=str,
                        choices=['mnist', 'fmnist', 'cifar10', 'celeba', 'nhl'],
                        default='cifar10')
    parser.add_argument('--epochs', '-e', type=int, default=100)
    parser.add_argument('--image_size', '-s', type=int, default=32)
    parser.add_argument('--n_colors', '-c', type=int, default=3)
    parser.add_argument('--batch_size', type=int,
                        default=32)  # DCGAN paper original value used 128 (32 is generally better to prevent vanishing gradients with SGAN and LSGAN, not important with relativistic GANs)
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='Adam betas[0], DCGAN paper recommends .50 instead of the usual .90')
    parser.add_argument('--beta2', type=float, default=0.999, help='Adam betas[1]')
    parser.add_argument('--decay', type=float, default=0,
                        help='Decay to apply to lr each cycle. decay^n_iter gives the final lr. Ex: .00002 will lead to .13 of lr after 100k cycles')
    parser.add_argument('--SELU', type=bool, default=False,
                        help='Using scaled exponential linear units (SELU) which are self-normalizing instead of ReLU with BatchNorm. Used only in arch=0. This improves stability.')
    parser.add_argument("--NN_conv", type=bool, default=False,
                        help="This approach minimize checkerboard artifacts during training. Used only by arch=0. Uses nearest-neighbor resized convolutions instead of strided convolutions (https://distill.pub/2016/deconv-checkerboard/ and github.com/abhiskk/fast-neural-style).")
    parser.add_argument('--seed', type=int)
    parser.add_argument('--input_folder', default='Datasets/Meow_64x64', help='input folder')
    parser.add_argument('--output_folder', default='Output/GANlosses', help='output folder')
    parser.add_argument('--inception_folder', default='Inception',
                        help='Inception model folder (path must exists already, model will be downloaded automatically)')
    parser.add_argument('--load', default=None,
                        help='Full path to network state to load (ex: /home/output_folder/run-5/models/state_11.pth)')
    parser.add_argument('--cuda', type=bool, default=True, help='enables cuda')
    parser.add_argument('--n_gpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--penalty', type=float, default=10, help='Gradient penalty parameter for WGAN-GP')
    parser.add_argument('--spectral', type=bool, default=False,
                        help='If True, use spectral normalization to make the discriminator Lipschitz. This Will also remove batch norm in the discriminator.')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='L2 regularization weight. Helps convergence but leads to artifacts in images, not recommended.')
    parser.add_argument('--gen_extra_images', type=int, default=50000,
                        help='Generate additional images with random fake cats in calculating FID (Recommended to use the same amount as the size of the dataset; for CIFAR-10 we use 50k, but most people use 10k) It must be a multiple of 100.')
    parser.add_argument('--gen_every', type=int, default=100000,
                        help='Generate additional images with random fake cats every x iterations. Used in calculating FID.')
    parser.add_argument('--extra_folder', default='./Output/Extra',
                        help='Folder for extra photos (different so that my dropbox does not get overwhelmed with 50k pictures)')
    parser.add_argument('--show_graph', type=bool, default=False,
                        help='If True, show gradients graph. Really neat for debugging.')
    parser.add_argument('--Tanh_GD', type=bool, default=False, help='If True, tanh everywhere.')
    parser.add_argument('--grad_penalty', type=bool, default=False,
                        help='If True, use gradient penalty of WGAN-GP but with whichever loss_D chosen. No need to set this true with WGAN-GP.')
    parser.add_argument('--print_every', type=int, default=1000,
                        help='Generate a mini-batch of images at every x iterations (to see how the training progress, you can do it often).')
    parser.add_argument('--save', type=bool, default=True,
                        help='Do we save models, yes or no? It will be saved in extra_folder')
    parser.add_argument('--cuda_device', type=str, choices=['cuda:0', 'cuda:1'], default='cuda:0')

    ######################
    # Generator parameters
    ######################
    parser.add_argument('--z_size', type=int, default=128)
    parser.add_argument('--G_h_size', type=int, default=128,
                        help='Number of hidden nodes in the Generator. Used only in arch=0. Too small leads to bad results, too big blows up the GPU RAM.')  # DCGAN paper original value
    parser.add_argument('--lr_G', type=float, default=.0001, help='Generator learning rate')
    parser.add_argument('--Giters', type=int, default=1, help='Number of iterations of G.')
    parser.add_argument('--spectral_G', type=bool, default=False,
                        help='If True, use spectral normalization to make the generator Lipschitz (Generally only D is spectral, not G). This Will also remove batch norm in the discriminator.')
    parser.add_argument('--no_batch_norm_G', type=bool, default=False, help='If True, no batch norm in G.')

    ##########################
    # Discriminator parameters
    ##########################
    parser.add_argument('--lr_D', type=float, default=.0001, help='Discriminator learning rate')
    parser.add_argument('--D_h_size', type=int, default=128,
                        help='Number of hidden nodes in the Discriminator. Used only in arch=0. Too small leads to bad results, too big blows up the GPU RAM.')  # DCGAN paper original value
    parser.add_argument('--no_batch_norm_D', type=bool, default=False, help='If True, no batch norm in D.')
    parser.add_argument('--Diters', type=int, default=1, help='Number of iterations of D')

    arguments = parser.parse_args()

    conf = PrettyTable()
    conf.field_names = ["Parameters", "Values"]
    conf.add_row(["Method", arguments.loss_D])
    conf.add_row(["Dataset", arguments.dataset])
    conf.add_row(["Image size", arguments.image_size])
    conf.add_row(["Channels", arguments.n_colors])
    conf.add_row(["Batch size", arguments.batch_size])
    conf.add_row(["Epochs", arguments.epochs])
    print(conf)

    main(arguments)
