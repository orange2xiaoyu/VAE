from ConvVAE import ConvVAE
from PIL import Image
import torch
from torchvision import transforms

state = torch.load('./model/best_model_50')
model = ConvVAE()
model.load_state_dict(state)

transform = transforms.Compose([
    lambda x: Image.open(x).convert('RGB'),
    transforms.Resize([94,310]),
    transforms.ToTensor(),
])


with torch.no_grad():
    model.eval()
    image_path = "/home/jinyu/lijinyu/datasets/kitti/data_odometry/stereo/00/image_0/000000.png"
    image = transform(image_path).unsqueeze(0)
    vector_out = model.fc3(model.fc2(model.fc1((model.encoder(image)).view(1, -1))))
    print("vector_out.size is ",vector_out.size())         
