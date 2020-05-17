#For testing purpose only 
#using https://github.com/lukemelas/EfficientNet-PyTorch README instructions

import json
from PIL import Image

import torch
from torchvision import transforms

from efficientnet_pytorch import EfficientNet

# Preprocess image
tfms = transforms.Compose([transforms.Scale(224),
                           transforms.CenterCrop(224),
                           transforms.ToTensor(),
                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])

img = tfms(Image.open('img4.jpg')).unsqueeze(0) #img.jpg, img2.png, img3.jpeg, img4.jpg
# img = img.repeat(1, 3, 1, 1)
print(img.shape) # torch.Size([1, 3, 224, 224])

# Load ImageNet class names
labels_map = json.load(open('labels_map.txt'))
labels_map = [labels_map[str(i)] for i in range(1000)]

# Classify
model = EfficientNet.from_pretrained('efficientnet-b7')


#Train





#Detect
model.eval()
with torch.no_grad():
    logits = model(img) #tensor

preds = torch.topk(logits, k=5)[1].squeeze(0).tolist()

print(type(preds))

# Print predictions
print('-----')
for idx in preds:
    label = labels_map[idx]
    prob = torch.softmax(logits, dim=1)[0, idx].item()
    print('{:<75} ({:.2f}%)'.format(label, prob*100))