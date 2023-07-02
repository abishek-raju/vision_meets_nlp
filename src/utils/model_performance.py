import torch
import matplotlib.pyplot as plt
import numpy
import io
from PIL import Image


def get_correct_and_misclassified_images(model,test_loader,number_of_images = 20,device = "cpu"):
    """To get the correct and misclassified images for the given model
    """
    misclassified_imgs = []
    correct_classified_imgs = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            is_correct = pred.eq(target.view_as(pred))
            misclassified_inds = (is_correct==0).nonzero()[:,0]
            correct_classified_inds = (is_correct==1).nonzero()[:,0]
            for mis_ind in misclassified_inds:
                if len(misclassified_imgs) < number_of_images:
                    misclassified_imgs.append({
                        "target": target[mis_ind].cpu().numpy(),
                        "pred": pred[mis_ind][0].cpu().numpy(),
                        "img": data[mis_ind].cpu().numpy()
                    })
                else:
                    break
            for mis_ind in correct_classified_inds:
                if len(correct_classified_imgs) < number_of_images:
                    correct_classified_imgs.append({
                        "target": target[mis_ind].cpu().numpy(),
                        "pred": pred[mis_ind][0].cpu().numpy(),
                        "img": data[mis_ind].cpu().numpy()
                    })
                else:
                    break
    return correct_classified_imgs,misclassified_imgs

def get_correct_and_misclassified_images_grid(model,test_loader,number_of_images = 20,device = "cpu"):
    """To get the correct and misclassified images for the given model
    """
    correct_classified_imgs,misclassified_imgs = get_correct_and_misclassified_images(model,test_loader,number_of_images= 20,device= "cpu")
    figure = plt.figure(figsize=(10, 10))
    figure.suptitle(str(test_loader.dataset.class_to_idx) + '\n', fontsize=16)
    for index in range(1, number_of_images + 1):
        plt.subplot(int(numpy.ceil(number_of_images/4)),4, index)
        plt.axis('off')
        plt.imshow(correct_classified_imgs[index-1]["img"].transpose(1, 2, 0),interpolation='nearest')
        plt.title("\nPredicted: %s\nActual: %s " % (correct_classified_imgs[index-1]["pred"], correct_classified_imgs[index-1]["target"]))
    plt.tight_layout()
    
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')

    im = Image.open(img_buf)
    im.show(title="My Image")

    img_buf.close()
    correct_classified_grid = im
    
    figure = plt.figure(figsize=(10, 10))
    figure.suptitle(str(test_loader.dataset.class_to_idx) + '\n', fontsize=16)
    for index in range(1, number_of_images + 1):
        plt.subplot(int(numpy.ceil(number_of_images/4)),4, index)
        plt.axis('off')
        plt.imshow(misclassified_imgs[index-1]["img"].transpose(1, 2, 0),interpolation='nearest')
        plt.title("\nPredicted: %s\nActual: %s " % (misclassified_imgs[index-1]["pred"], misclassified_imgs[index-1]["target"]))
    plt.tight_layout()
    
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')

    im = Image.open(img_buf)
    im.show(title="My Image")

    img_buf.close()
    mis_classified_grid = im
    
    return numpy.transpose(numpy.asarray(correct_classified_grid)[:,:,:3],(2,0,1)),numpy.transpose(numpy.asarray(mis_classified_grid)[:,:,:3],(2,0,1))