from PIL import Image
import torchvision.transforms as transforms


def custom_pil_loader(path):
# open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        img.load()
        return img
basicTransform = transforms.Compose(
    [transforms.ToTensor(),
     #extractOneChannel(),
    #transforms.RandomCrop((30,6)),
    transforms.Resize((44,6)),
     transforms.Normalize((0.5), (0.5))]
)
randCropTransform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.RandomCrop((40,6)),
    transforms.Resize((44,6)),
     transforms.Normalize((0.5), (0.5))])