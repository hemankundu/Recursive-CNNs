''' Document Localization using Recursive CNN
 Maintainer : Khurram Javed
 Email : kjaved@ualberta.ca '''

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

import model


class corner_finder():
    def __init__(self, CHECKPOINT_DIR, model_type="resnet"):

        self.model = model.ModelFactory.get_model(model_type, "corner")
        self.model.load_state_dict(torch.load(CHECKPOINT_DIR, map_location='cpu'))
        if torch.cuda.is_available():
            self.model.cuda()
        self.model.eval()

    def get_location(self, img, retainFactor=0.85):
        with torch.no_grad():
 
            myImage = np.copy(img)

            test_transform = transforms.Compose([transforms.Resize([32, 32]),
                                                 transforms.ToTensor()])



            img_temp = Image.fromarray(myImage)
            img_temp = test_transform(img_temp)
            img_temp = img_temp.unsqueeze(0)

            if torch.cuda.is_available():
                img_temp = img_temp.cuda()
            response = self.model(img_temp).cpu().data.numpy()
            response = response[0]

            return response


if __name__ == "__main__":
    pass
