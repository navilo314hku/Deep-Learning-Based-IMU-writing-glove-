import torchvision.transforms.functional as TF

class extractOneChannel:
    """Rotate by one of the given angles."""
    def __call__(self, x):#x is a 3 channel input tensor
        print(x.size())
        x=x[0]
        return x
        
