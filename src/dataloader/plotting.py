import torch
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import torchvision
from . import transforms

def plot_disparity(left_image, right_image, gt_disp, pred_disp, accuracy, image_pair_path):
    pred_disp = pred_disp.cpu().squeeze(0)
    # left_image = left_image.cpu().squeeze(0)
    # right_image = right_image.cpu().squeeze(0)
    gt_disp = gt_disp.cpu().squeeze(0)
    
    # plotting original images from dataset
    # left_image = torchvision.io.read_image(image_pair_path[0]).float() 
    left_image = plt.imread(image_pair_path[0])

    plt.imshow(pred_disp,plt.set_cmap('gray'))
    plt.axis("off")
    plt.imsave("pred_150_10_tri.png", pred_disp)

    fig, original = plt.subplots(1, 2)
    original[0].imshow(left_image)
    original[0].axis("off")
    # original[0].set_title('Left image')
    # original[1].imshow(right_image.permute(1,2,0))
    # original[1].axis("off")
    # original[1].set_title('Right image')
    original[1].imshow(gt_disp,plt.set_cmap('gray'))
    original[1].axis("off")
    # original[2].set_title('GT Disparity')
    # original[3].imshow(pred_disp)
    # original[3].axis("off")
    # original[3].set_title('Pred Disparity, 3PE:%f'%accuracy)
    # fig.suptitle("Inputs and Output")
    plt.imsave("left.png", left_image)
    plt.imsave("gt.png", gt_disp)


def plot_aug(image_pair_path):
    transform = transforms.get_transforms()
    left_image = torchvision.io.read_image(image_pair_path[0]).float() / 255.0
    right_image = torchvision.io.read_image(image_pair_path[1]).float() / 255.0
    disparity = torch.from_numpy(np.array(Image.open(image_pair_path[2])) / 256).float().unsqueeze(0)
    # plot
    left_image_1 = transform(left_image)
    right_image_1= transform(right_image)
    left_image_2 = transform(left_image)
    right_image_2= transform(right_image)
    left_image_3 = transform(left_image)
    right_image_3= transform(right_image)
    fig_original, original = plt.subplots(1, 2)
    fig_augment, augment = plt.subplots(3, 2)

    original[0].imshow(left_image.permute(1,2,0))
    original[0].axis("off")
    original[1].imshow(right_image.permute(1,2,0))
    original[1].axis("off")

    augment[0, 0].imshow(left_image_1.permute(1,2,0))
    augment[0, 0].axis("off")
    augment[0, 1].imshow(right_image_1.permute(1,2,0))
    augment[0, 1].axis("off")
    augment[1, 0].imshow(left_image_2.permute(1,2,0))
    augment[1, 0].axis("off")
    augment[1, 1].imshow(right_image_2.permute(1,2,0))
    augment[1, 1].axis("off")
    augment[2, 0].imshow(left_image_3.permute(1,2,0))
    augment[2, 0].axis("off")
    augment[2, 1].imshow(right_image_3.permute(1,2,0))
    augment[2, 1].axis("off")
    fig_original.suptitle("Original Left and Right")
    fig_augment.suptitle("Augmented Left and Right")

def smooth(f, K=5):
    """ Smoothing a function using a low-pass filter (mean) of size K """
    kernel = np.ones(K) / K
    f = np.concatenate([f[:int(K//2)], f, f[int(-K//2):]])  # to account for boundaries
    smooth_f = np.convolve(f, kernel, mode="same")
    smooth_f = smooth_f[K//2: -K//2]  # removing boundary-fixes
    return smooth_f


def load_pre_train_model(path_to_preTrainModel):
    checkpoint = torch.load(path_to_preTrainModel)
    stats = checkpoint['stats']
    return  stats

def visualize_progress(train_loss, val_loss, accuracy, start=0):
    # """ Visualizing loss and accuracy """
    # fig, ax = plt.subplots(2,2)
    # # fig.set_size_inches(24,5)

    # smooth_train = smooth(train_loss, 31)
    # ax[0, 0].plot(train_loss, c="blue", label="Loss", linewidth=3, alpha=0.5)
    # ax[0, 0].legend(loc="best")
    # ax[0, 0].set_xlabel("epoch")
    # ax[0, 0].set_ylabel("Mean Loss SL1")
    # ax[0, 0].set_yscale("linear")
    # ax[0, 0].set_title("Training Progress (linear)")
    
    # ax[0, 1].plot(train_loss, c="blue", label="Loss", linewidth=3, alpha=0.5)
    # ax[0, 1].legend(loc="best")
    # ax[0, 1].set_xlabel("epoch")
    # ax[0, 1].set_ylabel("Mean Loss SL1")
    # ax[0, 1].set_yscale("log")
    # ax[0, 1].set_title("Training Progress (log)")

    # smooth_val = smooth(val_loss, 31)
    # N_ITERS = len(val_loss)
    # ax[1, 0].plot(np.arange(start, N_ITERS)+start, val_loss[start:], c="blue", label="Loss", linewidth=3, alpha=0.5)
    # ax[1, 0].legend(loc="best")
    # ax[1, 0].set_xlabel("epoch")
    # ax[1, 0].set_ylabel("Mean Loss SL1")
    # ax[1, 0].set_yscale("log")
    # ax[1, 0].set_title(f"Valid Progress")

    # smooth_acc = smooth(accuracy, 31)
    # N_ITERS = len(accuracy)
    # ax[1, 1].plot(np.arange(start, N_ITERS)+start, accuracy[start:], c="blue", label="Loss", linewidth=3, alpha=0.5)
    # ax[1, 1].legend(loc="best")
    # ax[1, 1].set_xlabel("epoch")
    # ax[1, 1].set_ylabel("Mean Loss 3PE")
    # ax[1, 1].set_yscale("linear")
    # ax[1, 1].set_title(f"Accuracy Progress")
    
    # plotting just loss and valid 
    fig, ax = plt.subplots(1,2)
    smooth_train = smooth(train_loss, 31)
    ax[0].plot(train_loss, c="blue", label="Loss", linewidth=3, alpha=0.5)
    ax[0].legend(loc="best")
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("Mean Loss SL1")
    ax[0].set_yscale("linear")
    ax[0].set_title("Training Progress")
    
    smooth_val = smooth(val_loss, 31)
    N_ITERS = len(val_loss)
    ax[1].plot(np.arange(start, N_ITERS)+start, val_loss[start:], c="red", label="Loss", linewidth=3, alpha=0.5)
    ax[1].legend(loc="best")
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("Mean Loss SL1")
    ax[1].set_yscale("log")
    ax[1].set_title(f"Valid Progress")
    plt.tight_layout()
    return
