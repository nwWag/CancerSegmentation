import torch
import torch.functional as F
import numpy as np

from tqdm import tqdm
from colorama import Fore
from PIL import Image
from skimage import data, color, io, img_as_float


def test(model, test_dataset, loss_f, batch_size=48, shuffle=False, num_workers=0, collate_fn=None):
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                              shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)

    tkb = tqdm(total=int(len(test_loader)), bar_format="{l_bar}%s{bar}%s{r_bar}" % (
        Fore.GREEN, Fore.RESET), desc="Test Pixel IOU ")

    loss_sum = 0
    for batch_id, data in enumerate(test_loader):
        if data[0].shape[0] != test_loader.batch_size:
            continue

        inputs, labels = data
        with torch.no_grad():
            outputs = model(inputs.to('cuda'))[0]
            loss = loss_f(outputs, labels.to('cuda'))
            loss_sum += loss.item()
        tkb.set_postfix(Accuracy='{:3f}'.format(
            loss_sum / (batch_id+1)))
        tkb.update(1)


def draw(model, example, path_to_store='example/'):
    example_img, example_label_raw = example[0], example[1]
    example_label = (example_label_raw == 0.0).squeeze(0).to('cuda')
    with torch.no_grad():
        outputs, scores = model(example_img.unsqueeze(0).to('cuda'))
    predicted = (torch.argmax(outputs, dim=1) == 0.0).squeeze(0)
    gray_img = (example_img[0] * 0.2125 +
                example_img[1] * 0.7154 + example_img[2] * 0.0721).repeat(3, 1, 1).permute(1, 2, 0)

    example_img_p = example_img.clone().permute(1, 2, 0)
    example_img_o = example_img_p.clone()
    example_img_a = torch.cat((scores[0].repeat(2,1,1), torch.zeros_like(scores[0])), dim=0).permute(1,2,0) * 255.0

    example_img_p[predicted] = gray_img[predicted]
    example_img_o[example_label] = gray_img[example_label]

    im = Image.fromarray(example_img_o.cpu().numpy().astype(np.uint8))
    im.save(path_to_store + 'org.png')

    im = Image.fromarray(example_img_p.cpu().numpy().astype(np.uint8))
    im.save(path_to_store + 'pred.png')


    im_clean = Image.fromarray(example_img.permute(1,2,0).cpu().numpy().astype(np.uint8)).convert('LA').convert('RGBA')
    im_a = Image.fromarray(example_img_a.cpu().numpy().astype(np.uint8)).convert('RGBA')
    im_mask = Image.new('RGBA',im_clean.size,(0,0,0,100))
    im = Image.composite(im_clean, im_a, im_mask).convert('RGB')
    #im = Image.fromarray(img_masked)
    im.save(path_to_store + 'att.png')

    print("Written to", path_to_store)


def load(model, name="store/base"):
    """Loads model parameters from disk.

    Args:
        name (str, optional): Path to storage. Defaults to "store/base".
    """
    pretrained_dict = torch.load(name + ".pt")
    print("Loaded", name + " model.")
    model_dict = model.state_dict()
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
