import os

folder = 'data/test/KITTI2012/original/colored_1'  # change this to your folder path

for filename in os.listdir(folder):
    if filename.endswith('.png'):
        base = filename[:-4]  # remove '.png'
        new_name = base + 'R.png'
        os.rename(os.path.join(folder, filename), os.path.join(folder, new_name))

print("Renaming complete.")
