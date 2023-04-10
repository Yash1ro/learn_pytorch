import MyLogs
from torchvision import transforms
from PIL import Image

img_path = "anime_image/103839408_p0.jpg"
img = Image.open(img_path)

tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)

# Normlize the tensor picture
tensor_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
norm_img = tensor_norm(tensor_img)

tensor_resize = transforms.Resize((512, 512))
resize_img = tensor_resize(tensor_img)

tensor_compose = transforms.Compose([tensor_trans, tensor_resize, tensor_norm])
comp_img = tensor_compose(img)

tb = MyLogs.Tensorboard()
writer = tb.create_board()
writer.add_image("To_Tensor", tensor_img, 1)
writer.add_image("Normalization", norm_img, 1)
writer.add_image("Resize", resize_img, 1)
writer.add_image("Compose", comp_img, 1)
writer.close()

