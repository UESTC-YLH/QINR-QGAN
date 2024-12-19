import os
import argparse
import math
import numpy as np
import torch
import struct
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision.utils import save_image
import torchvision.transforms as transforms
from utils.dataset import load_mnist, load_fmnist, denorm, select_from_dataset, DigitsDataset, load_Emnist
from utils.wgan import compute_gradient_penalty
# from models.QGCC import PQWGAN_CC
from models.QINR import PQWGAN_CC
from models.QGQC import PQWGAN_QC
from utils.compute_fid_kl import calculate_fid, calculate_cos, BoundarySeekingLoss
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

def train(classes_str, dataset_str, patches, layers, n_data_qubits, batch_size, out_folder, checkpoint, randn, patch_shape, qcritic):
    classes = list(set([int(digit) for digit in classes_str]))
    device = torch.device("cuda:1")
    n_epochs = 20
    image_size = 28  # 28
    channels = 1
    if dataset_str == "mnist":
        dataset = select_from_dataset(load_mnist(image_size=image_size), 500, classes)  # 1000
    elif dataset_str == "fmnist":
        dataset = select_from_dataset(load_fmnist(image_size=image_size), 500, classes)  # 1000
    elif dataset_str == "optdigits-data":
        image_size = 8
        classes = int(np.array(classes))
        transform = transforms.Compose([transforms.ToTensor()])
        # dataset = DigitsDataset(csv_file="./datasets/optdigits-data/optdigits.tra", transform=transform)
        dataset = DigitsDataset(csv_file="./datasets/optdigits-data/optdigits.tra", transform=transform, label=classes)
    elif dataset_str == "Emnist":
        dataset = select_from_dataset(load_Emnist(image_size=image_size), 500, classes)  # 1000
    ancillas = 1
    if n_data_qubits:
        qubits = n_data_qubits + ancillas
    else:
        qubits = math.ceil(math.log(image_size ** 2 // patches, 2)) + ancillas

    if qcritic:
        lr_D = 0.01
    else:
        lr_D = 0.00012#0.0002
    lr_G = 0.00001#0.01
    b1 = 0
    b2 = 0.9
    latent_dim = qubits
    lambda_gp = 10
    n_critic = 1
    n_gen_step = 1
    sample_interval = 10
    num_iterations = 2000
    all_fid = []
    all_KL = []
    all_d_loss = []
    all_g_loss = []
    all_ssim = []
    all_cos_sim = []
    out_dir = f"{out_folder}/dig_num_{classes_str}_patch_num_{patches}_layers_num_{layers}_batch_size_num_{batch_size}_gen_K_{n_gen_step}_{dataset_str}_lr_G_{lr_G}_lr_D_{lr_D}"
    if randn:
        out_dir += "_randn"
    if patch_shape[0] and patch_shape[1]:
        out_dir += f"_{patch_shape[0]}x{patch_shape[1]}ps"

    os.makedirs(out_dir, exist_ok=True)
    if qcritic:
        gan = PQWGAN_QC(image_size=image_size, channels=channels, n_generators=patches, n_gen_qubits=qubits,n_ancillas=ancillas, n_gen_layers=layers, patch_shape=patch_shape, n_critic_qubits=10, n_critic_layers=175)
    else:
        # gan = PQWGAN_CC(image_size=image_size, channels=channels, n_generators=patches, n_qubits=qubits, n_ancillas=ancillas, n_layers=layers, patch_shape=patch_shape)
        gan = PQWGAN_CC(image_size=28, channels=1, in_features=args.in_features, out_features=1, hidden_features=args.hidden_features, hidden_layers=args.hidden_layers, spectrum_layer=args.spectrum_layer, use_noise=args.use_noise)

    critic = gan.critic.to(device)
    generator = gan.generator.to(device)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    optimizer_G = Adam(generator.parameters(), lr=lr_G, betas=(b1, b2))
    optimizer_C = Adam(critic.parameters(), lr=lr_D, betas=(b1, b2))
    if randn:
        fixed_z = torch.randn(batch_size, latent_dim, device=device)
    else:
        fixed_z = torch.rand(batch_size, latent_dim, device=device)

    wasserstein_distance_history = []
    saved_initial = False
    batches_done = 0

    if checkpoint != 0:
        # critic.load_state_dict(torch.load(out_dir + f"/critic-{checkpoint}.pt"))
        # generator.load_state_dict(torch.load(out_dir + f"/generator-{checkpoint}.pt"))
        wasserstein_distance_history = list(np.load(out_dir + "/wasserstein_distance.npy"))
        saved_initial = True
        batches_done = checkpoint
    counter = 0
    train_data = []
    classes = int(np.array(classes))
    for (data, labels) in dataloader:
        for x, y in zip(data, labels):
            if y == classes:
                # a = x.numpy().reshape([1, 64])
                train_data.append(x.numpy().reshape([1, 784]))
    train_data_new = np.array(train_data)
    new_train_data = np.squeeze(train_data_new, axis=1)[:500, :]
    # tensor_new_train_data = torch.tensor(new_train_data)
    # train_data_new
    while True:
        for i, (real_images, _) in enumerate(dataloader):
            if not saved_initial:
                fixed_images = generator(fixed_z)
                # KL = torch.nn.functional.kl_div(fixed_images.reshape([1, 64]).softmax(-1).log(), real_images.reshape([1, 64]).softmax(-1), reduction='sum')
                save_image(denorm(fixed_images), os.path.join(out_dir, '{}.png'.format(batches_done)), nrow=5)
                save_image(denorm(real_images), os.path.join(out_dir, 'real_samples.png'), nrow=5)
                saved_initial = True

            real_images = real_images.to(device)
            optimizer_C.zero_grad()

            if randn:
                z = torch.randn(batch_size, latent_dim, device=device)
            else:
                z = torch.rand(batch_size, latent_dim, device=device)
            fake_images = generator(z)

            # Real images
            real_validity = critic(real_images)
            # Fake images
            fake_validity = critic(fake_images)
            # Gradient penalty 梯度惩罚
            gradient_penalty = compute_gradient_penalty(critic, real_images, fake_images, device)
            # Adversarial loss
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty
            wasserstein_distance = torch.mean(real_validity) - torch.mean(fake_validity)
            wasserstein_distance_history.append(wasserstein_distance.item())

            d_loss.backward()
            optimizer_C.step()

            optimizer_G.zero_grad()

            # Train the generator every n_critic steps
            for jj in range(n_gen_step):
                # print(jj)
            # if i % n_critic == 0:

                # -----------------...............................................................................
                #  Train Generator
                # -----------------

                # Generate a batch of images
                fake_images = generator(z)
                # Loss measures generator's ability to fool the discriminator
                # Train on fake images
                fake_validity = critic(fake_images)
                g_loss = -torch.mean(fake_validity)

                g_loss.backward()
                optimizer_G.step()

                print(f"[[Batch {i}/{len(dataloader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}] [Wasserstein Distance: {wasserstein_distance.item()}]")
                np.save(os.path.join(out_dir, 'wasserstein_distance.npy'), wasserstein_distance_history)
                batches_done += n_critic

                if batches_done % sample_interval == 0:
                    fixed_images = generator(fixed_z)
                    save_image(denorm(fixed_images), os.path.join(out_dir, '{}.png'.format(batches_done)), nrow=5)
                    # torch.save(critic.state_dict(), os.path.join(out_dir, 'critic-{}.pt'.format(batches_done)))
                    # torch.save(generator.state_dict(), os.path.join(out_dir, 'generator-{}.pt'.format(batches_done)))
                    print("saved images and state")
            counter = counter + 1
            print(f'Iteration: {counter}')
            fid_fixed_images = generator(fixed_z)
            fid_real_images = new_train_data
            fid = calculate_fid(fid_fixed_images, fid_real_images)
            cos_sim = calculate_cos(fid_fixed_images, real_images)
            a_ssim = fid_fixed_images.detach().cpu().numpy().reshape(image_size, image_size)
            b_ssim = real_images.detach().cpu().numpy().reshape(image_size, image_size)
            ssim = structural_similarity(a_ssim, b_ssim, data_range=1)
            psnr = peak_signal_noise_ratio(b_ssim, a_ssim)
            all_ssim.append(ssim)
            all_fid.append(fid)
            # kl_data = torch.tensor(new_train_data)
            # KL = torch.nn.functional.kl_div(fixed_images.reshape([1, 784]).softmax(-1).log(), kl_data.softmax(-1), reduction='batchmean')
            all_KL.append(psnr)
            all_g_loss.append(g_loss.detach().cpu().numpy())
            all_d_loss.append(d_loss.detach().cpu().numpy())
            all_cos_sim.append(cos_sim)
            if counter == num_iterations:
                break
        if counter == num_iterations:
            all_KL = np.array(all_KL)
            np.save(f'{out_dir}/FID_{classes_str}', all_fid)
            np.save(f'{out_dir}/KL_{classes_str}', all_KL)
            np.save(f'{out_dir}/g_loss_{classes_str}', all_g_loss)
            np.save(f'{out_dir}/d_loss_{classes_str}', all_d_loss)
            np.save(f'{out_dir}/cos_sim_{classes_str}', all_cos_sim)
            np.save(f'{out_dir}/ssim_{classes_str}', all_ssim)
            break



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-cl0", "--classes0", help="classes to train on", default=[0], type=str)
    parser.add_argument("-cl1", "--classes1", help="classes to train on", default=[1], type=str)
    parser.add_argument("-cl2", "--classes2", help="classes to train on", default=[2], type=str)
    parser.add_argument("-cl3", "--classes3", help="classes to train on", default=[3], type=str)
    parser.add_argument("-cl4", "--classes4", help="classes to train on", default=[4], type=str)
    parser.add_argument("-cl5", "--classes5", help="classes to train on", default=[5], type=str)
    parser.add_argument("-cl6", "--classes6", help="classes to train on", default=[6], type=str)
    parser.add_argument("-cl7", "--classes7", help="classes to train on", default=[7], type=str)
    parser.add_argument("-cl8", "--classes8", help="classes to train on", default=[8], type=str)
    parser.add_argument("-cl9", "--classes9", help="classes to train on", default=[9], type=str)

    parser.add_argument("-d", "--dataset", help="dataset to train on", default="mnist", type=str)#optdigits-data,mnist
    parser.add_argument("-p", "--patches", help="number of sub-generators", default=4, type=int, choices=[1, 2, 4, 7, 14, 28])
    parser.add_argument("-l", "--layers", help="layers per sub-generators", default=2, type=int)
    parser.add_argument("-q", "--qubits", help="number of data qubits per sub-generator", type=int, default=5)
    parser.add_argument("-b", "--batch_size", help="batch_size", default=1, type=int)
    parser.add_argument("-o", "--out_folder", help="output directory", default='./My-Enhanced-resutls', type=str)
    parser.add_argument("-c", "--checkpoint", help="checkpoint to load from", type=int, default=0)
    parser.add_argument("-rn", "--randn", help="use normal prior, otherwise use uniform prior", action="store_true")
    parser.add_argument("-ps", "--patch_shape", help="shape of sub-generator output (H, W)", default=[None, None], type=int, nargs=2)
    parser.add_argument("-qc", "--qcritic", help="use quantum critic", action="store_true")


    parser.add_argument('--in_features', type=int, default=6) #in_features=1 sound; in_features=2 image
    parser.add_argument('--hidden_features', type=int, default=6)
    parser.add_argument('--hidden_layers', type=int, default=2)
    # parser.add_argument('--first_omega_0', type=float, default=1)
    # parser.add_argument('--hidden_omega_0', type=float, default=1)
    parser.add_argument('--spectrum_layer', type=int, default=2)
    parser.add_argument('--use_noise', type=float, default=0)
    args = parser.parse_args()

    # train(args.classes, args.dataset, args.patches, args.layers, args.qubits, args.batch_size, args.out_folder, args.checkpoint, args.randn, tuple(args.patch_shape), args.qcritic)
    train(args.classes0, args.dataset, args.patches, args.layers, args.qubits, args.batch_size, args.out_folder, args.checkpoint, args.randn, tuple(args.patch_shape), args.qcritic)
    print("0 finsih")
    train(args.classes1, args.dataset, args.patches, args.layers, args.qubits, args.batch_size, args.out_folder, args.checkpoint, args.randn, tuple(args.patch_shape), args.qcritic)
    print("1 finsih")
    train(args.classes2, args.dataset, args.patches, args.layers, args.qubits, args.batch_size, args.out_folder, args.checkpoint, args.randn, tuple(args.patch_shape), args.qcritic)
    print("2 finsih")
    train(args.classes3, args.dataset, args.patches, args.layers, args.qubits, args.batch_size, args.out_folder, args.checkpoint, args.randn, tuple(args.patch_shape), args.qcritic)
    print("3 finsih")
    train(args.classes4, args.dataset, args.patches, args.layers, args.qubits, args.batch_size, args.out_folder, args.checkpoint, args.randn, tuple(args.patch_shape), args.qcritic)
    print("4 finsih")
    train(args.classes5, args.dataset, args.patches, args.layers, args.qubits, args.batch_size, args.out_folder, args.checkpoint, args.randn, tuple(args.patch_shape), args.qcritic)
    print("5 finsih")
    train(args.classes6, args.dataset, args.patches, args.layers, args.qubits, args.batch_size, args.out_folder, args.checkpoint, args.randn, tuple(args.patch_shape), args.qcritic)
    print("6 finsih")
    train(args.classes7, args.dataset, args.patches, args.layers, args.qubits, args.batch_size, args.out_folder, args.checkpoint, args.randn, tuple(args.patch_shape), args.qcritic)
    print("7 finsih")
    train(args.classes8, args.dataset, args.patches, args.layers, args.qubits, args.batch_size, args.out_folder, args.checkpoint, args.randn, tuple(args.patch_shape), args.qcritic)
    print("8 finsih")
    train(args.classes9, args.dataset, args.patches, args.layers, args.qubits, args.batch_size, args.out_folder, args.checkpoint, args.randn, tuple(args.patch_shape), args.qcritic)
    print("9 finsih")
