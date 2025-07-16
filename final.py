# !unzip /content/CRC-VAL-HE-7K.zip

# Install required packages
# !pip install torchcam pytorch_lightning transformers datasets evaluate

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
import pytorch_lightning as pl
from transformers import ViTModel
from sklearn.metrics import accuracy_score, cohen_kappa_score, roc_auc_score

binary_map = {'ADI':0,'BACK':0,'DEB':0,'LYM':0,'MUC':0,'MUS':0,'NORM':0,'STR':1,'TUM':1}

# Adjust dataset_dir path as needed
dataset_dir = '/content/CRC-VAL-HE-7K'
rows = []

for cls in os.listdir(dataset_dir):
    p = os.path.join(dataset_dir, cls)
    if os.path.isdir(p) and cls in binary_map:
        for f in os.listdir(p):
            if f.lower().endswith(('.tif','.png','.jpg')):
                rows.append([os.path.join(p, f), binary_map[cls], cls])

df = pd.DataFrame(rows, columns=['filepath','label','tissue_class'])
df.to_csv('colon_binary_labels.csv', index=False)
print(df['label'].value_counts())

df = pd.read_csv('colon_binary_labels.csv')
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

class ColonDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row.filepath).convert('RGB')
        if self.transform: img = self.transform(img)
        return {'pixel_values': img, 'labels': torch.tensor(int(row.label))}

train_ds = ColonDataset(train_df, transform)
val_ds = ColonDataset(val_df, transform)

w = 1.0 / np.bincount(train_df['label'])
sam = WeightedRandomSampler(w[train_df['label']], len(train_df))

train_loader = DataLoader(train_ds, batch_size=32, sampler=sam, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)

class CNN_ViT_Hybrid(pl.LightningModule):
    def __init__(self, lr=5e-5):
        super().__init__()
        self.save_hyperparameters()
        cnn = models.resnet50(weights="IMAGENET1K_V1")
        self.cnn = nn.Sequential(*list(cnn.children())[:-2])
        self.conv1x1 = nn.Conv2d(2048, 3, kernel_size=1)
        self.up = nn.Upsample((224,224), mode='bilinear', align_corners=False)
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.fc = nn.Linear(self.vit.config.hidden_size, 2)
        self.ce = nn.CrossEntropyLoss()

    def forward(self, x):
        f = self.cnn(x)
        f = self.up(self.conv1x1(f))
        s = self.vit(pixel_values=f).last_hidden_state[:,0,:]
        return self.fc(s)

    def training_step(self, b,_):
        logits = self(b['pixel_values'])
        loss = self.ce(logits, b['labels'])
        acc = (logits.argmax(1)==b['labels']).float().mean()
        self.log('train_loss', loss); self.log('train_acc', acc)
        return loss

    def configure_optimizers(self): return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)

from sklearn.metrics import accuracy_score, cohen_kappa_score, roc_auc_score

def eval_model(model, loader):
    model.eval()
    preds, probs, labs = [], [], []
    with torch.no_grad():
        for b in loader:
            logits = model(b['pixel_values'])
            probs.extend(torch.softmax(logits,1)[:,1].cpu().tolist())
            p = logits.argmax(1).cpu().tolist()
            preds.extend(p); labs.extend(b['labels'].cpu().tolist())
    return accuracy_score(labs, preds), cohen_kappa_score(labs, preds), roc_auc_score(labs, probs)



# For hybrid:
model2 = CNN_ViT_Hybrid()
trainer = pl.Trainer(max_epochs=2, accelerator='gpu' if torch.cuda.is_available() else 'cpu')
trainer.fit(model2, train_loader, val_loader)

acc2, kappa2, auc2 = eval_model(model2, val_loader)
print("CNN–ViT Hybrid → Acc: {:.4f}, Kappa: {:.4f}, AUC: {:.4f}".format(acc2, kappa2, auc2))

print("CNN–ViT Hybrid → Acc: {:.4f}, Kappa: {:.4f}, AUC: {:.4f}".format(acc2, kappa2, auc2))


class ResNetClassifier(pl.LightningModule):
    def __init__(self, lr=5e-5):
        super().__init__()
        self.save_hyperparameters()
        self.model = models.resnet50(weights="IMAGENET1K_V1")
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)
        self.ce = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, b, _):
        logits = self(b['pixel_values'])
        loss = self.ce(logits, b['labels'])
        acc = (logits.argmax(1) == b['labels']).float().mean()
        self.log('train_loss', loss); self.log('train_acc', acc)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)

# Train and evaluate ResNet model
model1 = ResNetClassifier()
trainer = pl.Trainer(max_epochs=1, accelerator='gpu' if torch.cuda.is_available() else 'cpu')
trainer.fit(model1, train_loader, val_loader)

acc1, kappa1, auc1 = eval_model(model1, val_loader)
print("ResNet50 → Acc: {:.4f}, Kappa: {:.4f}, AUC: {:.4f}".format(acc1, kappa1, auc1))