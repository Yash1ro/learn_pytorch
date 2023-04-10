import os

root_dir = "data/hymenoptera_data/hymenoptera_data/train"
target_dir = "ants_image"

img_path = os.listdir(os.path.join(root_dir, target_dir))
label = target_dir.split("_")[0]
out_dir = "ants_label"
for i in img_path:
    filename = i.split(".jpg")[0]
    with open(os.path.join(root_dir, out_dir, "{}.txt".format(filename)), 'w') as f:
        f.write(label)
