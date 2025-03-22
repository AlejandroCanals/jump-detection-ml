import os 

folders = [
    "../data/jump_segments",
    "../data/non_jump_segments",
    "../data/segments",
    "../data/augmented_data",
]

for folder in folders:
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

