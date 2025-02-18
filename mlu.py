import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
import os
import torch.backends.cudnn as cudnn
import pydicom
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import cv2
import torchio as tio
import torch.nn as nn
from torchvision.models.video import r3d_18




def train_model(model, dataloader, num_epochs=25):
    model.train()
    for epoch in range(start_epoch, num_epochs):  # Start from saved epoch
        running_loss = 0.0
        correct = 0
        total = 0

        for data in dataloader:
            # Check tensor shape and add channel if necessary
            inputs = data['image']

            if inputs.dim() == 4:  # Shape (B, D, H, W) to (B, 1, D, H, W)
                inputs = inputs.unsqueeze(1)
            elif inputs.dim() != 5 or inputs.shape[1] != 1:
                raise ValueError(f"Unexpected input shape: {inputs.shape}")

            # Transfer to device
            inputs = inputs.to(device, dtype=torch.float32)  # Ensure correct dtype
            labels = data['labels'].to(device, dtype=torch.long)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()

        accuracy = 100 * correct / total
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss:.4f}, Accuracy: {accuracy:.2f}%')

        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, checkpoint_path)

        if accuracy >= 90:
            print("Stopping early as accuracy reached 90%")
            break
        scheduler.step()

    # Clearing unused GPU memory after training
    torch.cuda.empty_cache()


def evaluate_model(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data['image'].to(device), data['label'].to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    precision = precision_score(all_labels, all_preds, average='binary', pos_label='M')
    recall = recall_score(all_labels, all_preds, average='binary', pos_label='M')
    f1 = f1_score(all_labels, all_preds, average='binary', pos_label='M')

    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F-score: {f1:.4f}')




# Custom Dataset class to handle DICOM images, masks, and metadata
class MedicalImageDataset(Dataset):
    def __init__(self, image_dir, mask_dir, metadata_img_file, metadata_mask_file, bbx_file, labels_file, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.metadata_img = pd.read_json(metadata_img_file)
        self.metadata_mask = pd.read_json(metadata_mask_file)
        self.bbx_labels = pd.read_csv(bbx_file)
        self.labels = pd.read_csv(labels_file)
        self.transform = transform

    def __len__(self):
        return len(self.metadata_img)


    def __getitem__(self, idx):
        # Load the patient number from metadata using the idx
        patient_number = f"{self.metadata_img.iloc[idx]['patient_number']:03d}"

        # Updated paths to match the correct directory and file format
        img_subdir = self.metadata_img.iloc[idx]['path'][:41]
        mask_subdir = os.path.join(self.mask_dir, patient_number)

        # Read and resize 3D image and mask volumes
        images = []
        masks = []

        for f in sorted(os.listdir(img_subdir)):
            img_slice = pydicom.dcmread(os.path.join(img_subdir, f)).pixel_array
            img_resized = cv2.resize(img_slice, (256, 256), interpolation=cv2.INTER_LINEAR)
            images.append(img_resized)

        for f in sorted(os.listdir(mask_subdir)):
            mask_slice = pydicom.dcmread(os.path.join(mask_subdir, f)).pixel_array
            mask_resized = cv2.resize(mask_slice, (256, 256), interpolation=cv2.INTER_NEAREST)
            masks.append(mask_resized)

        # Convert lists to numpy arrays and adjust depth to 300
        images = np.stack(images, axis=0)
        masks = np.stack(masks, axis=0)

        # Standardize depth to 250 slices
        target_depth = 250

        # Trim slices if they exceed target depth
        if images.shape[0] > target_depth:
            images = images[50:target_depth+50]
            masks = masks[50:target_depth+50]
        # Pad slices with zeros if they are less than target depth
        elif images.shape[0] < target_depth:
            pad_size = target_depth - images.shape[0]
            images = np.pad(images, ((0, pad_size), (0, 0), (0, 0)), mode='constant', constant_values=0)
            masks = np.pad(masks, ((0, pad_size), (0, 0), (0, 0)), mode='constant', constant_values=0)

        # Convert images and masks to float32
        images = images.astype(np.float32)
        masks = masks.astype(np.float32)

        # Load bounding box information
        bbx_data = self.bbx_labels[self.bbx_labels['id'] == int(patient_number)]
        bbx_list = [
            [row['c_x'], row['c_y'], row['c_z'], row['len_x'], row['len_y'], row['len_z'],
             int(self.labels[self.labels['case_id'] == int(patient_number)]['label'].iloc[0] == 'M')]
            for _, row in bbx_data.iterrows()
        ]

        # Get label for classification
        label = 1 if self.labels[self.labels['case_id'] == int(patient_number)]['label'].iloc[0] == 'M' else 0

        # Create sample dictionary
        sample = {
            'image': images,    # 3D image volume of shape (300, 512, 512)
            'mask': masks,      # 3D mask volume of shape (300, 512, 512)
            'bboxes': bbx_list, # Bounding box coordinates
            'labels': label     # Binary label
        }



        # Apply transformations if specified
        if self.transform:
            # Wrap the data in a TorchIO Subject
            subject = tio.Subject(
                image=tio.ScalarImage(tensor=images_np[None]),  # Add a channel dimension
                mask=tio.LabelMap(tensor=masks_np[None]),        # Add a channel dimension
                bboxes=tio.LabelMap(tensor=np.array(bbx_list)[None]),  # Wrap bounding boxes if needed
            )

            # Apply transformations
            transformed = self.transform(subject)

            # Extract the augmented image and mask
            sample['image'] = transformed.image[0]  # Get the transformed image tensor
            sample['mask'] = transformed.mask[0]      # Get the transformed mask tensor

            # Update bounding boxes after transformation
            # Extract updated bounding boxes from transformed subject if needed
            sample['bboxes'] = transformed.bboxes  # Make sure this handles the updated bounding boxes

        return sample


#--------------------------------------------------------------------------------------



os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

cudnn.benchmark = True

# Directory and file pathsf
image_dir = "/home/lmc6/USML/ABUS/DATA_Dicom_00_99"
mask_dir = "/home/lmc6/USML/ABUS/MASK_Dicom"
metadata_img_file = "/home/lmc6/USML/ABUS/output_file.json"
metadata_mask_file = "/home/lmc6/USML/ABUS/annotations.json"
bbx_file = "/home/lmc6/USML/ABUS/bbx_labels.csv"
labels_file = "/home/lmc6/USML/ABUS/labels.csv"

# Dataset and DataLoader
print("dataset start")
dataset = MedicalImageDataset(image_dir, mask_dir, metadata_img_file, metadata_mask_file, bbx_file, labels_file)

print("dataloader start")
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)

device = torch.device("cuda")

print("dataloader transfer start")
for batch in dataloader:
    inputs, labels = batch  # Unpack batch (adjust if using a dict or different structure)
    inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
# Define the augmentations
print("transform start")
transform = tio.Compose([
    tio.RescaleIntensity((0, 1)),  # Normalize the intensity
    tio.Resample((512, 512, 512)),  # Resize to (512, 512, 512), assuming you want to resize the entire volume
    tio.RandomFlip(axes=(0,)),      # Horizontal flip along the depth axis
    tio.RandomFlip(axes=(1,)),      # Vertical flip along the height axis
    #tio.RandomRotate(angles=(0, 180)),  # Randomly rotate the volume
    #tio.ShiftScaleRotate(),
    #tio.RandomResizedCrop((512, 512, 512)),  # Randomly crop the volume
    # Add more augmentations if needed
])


# In[12]:


#get_ipython().system('pip install --upgrade torchvision')


# In[13]:


print("build model")


model = r3d_18(weights=None)  # No weights to focus on architecture

# Move the model to GPU if available

model = model.to(device)

# Modify stem to accept 1 channel input and output 64 channels
model.stem[0] = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
model.stem[1] = nn.BatchNorm3d(32)

# Adjust Layer 1 (Output: 32 channels)
model.layer1[0].conv1 = nn.Conv3d(32, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
model.layer1[0].bn1 = nn.BatchNorm3d(32)
model.layer1[0].conv2 = nn.Conv3d(32, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
model.layer1[0].bn2 = nn.BatchNorm3d(32)
model.layer1[0].downsample = nn.Sequential(
    nn.Conv3d(32, 32, kernel_size=1, stride=1, bias=False),
    nn.BatchNorm3d(32)
)

# Adjust Layer 2 (Output: 64 channels)
model.layer2[0].conv1 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
model.layer2[0].bn1 = nn.BatchNorm3d(64)
model.layer2[0].conv2 = nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
model.layer2[0].bn2 = nn.BatchNorm3d(64)
model.layer2[0].downsample = nn.Sequential(
    nn.Conv3d(32, 64, kernel_size=1, stride=2, bias=False),
    nn.BatchNorm3d(64)
)

# Adjust Layer 3 (Output: 128 channels)
model.layer3[0].conv1 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
model.layer3[0].bn1 = nn.BatchNorm3d(128)
model.layer3[0].conv2 = nn.Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
model.layer3[0].bn2 = nn.BatchNorm3d(128)
model.layer3[0].downsample = nn.Sequential(
    nn.Conv3d(64, 128, kernel_size=1, stride=2, bias=False),
    nn.BatchNorm3d(128)
)

# Adjust Layer 4 (Output: 256 channels)
model.layer4[0].conv1 = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
model.layer4[0].bn1 = nn.BatchNorm3d(256)
model.layer4[0].conv2 = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
model.layer4[0].bn2 = nn.BatchNorm3d(256)
model.layer4[0].downsample = nn.Sequential(
    nn.Conv3d(128, 256, kernel_size=1, stride=2, bias=False),
    nn.BatchNorm3d(256)
)
# Update the classifier for binary classification
model.fc = nn.Linear(256, 2)


print(f"Model is running on device: {device}")
print(f"Using GPUs: {torch.cuda.device_count()}")

# This will tell you which device the model's parameters are on
for name, param in model.named_parameters():
    print(f"Parameter {name} is on device: {param.device}")
#Step 4 with a checkpoint
print("optimizer start")
# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Set checkpoint path
checkpoint_path = 'checkpoint.pth'

# Load checkpoint if available
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.module.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Resuming training from epoch {start_epoch}")
else:
    start_epoch = 0

# Training loop
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # For more accurate error tracking

print("start train")
train_model(model, inputs)

print("start evaluate")
evaluate_model(model, inputs)
