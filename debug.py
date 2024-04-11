import os

folder_path = "/home/student.aau.dk/saales20/local-global-Mul-Head-Attention/VIT_FSD50K/data_set/dev_22050"

folder_path1 = "/home/student.aau.dk/saales20/local-global-Mul-Head-Attention/VIT_FSD50K/data_set/eval_22050"


folder_path2 = "/home/student.aau.dk/saales20/local-global-Mul-Head-Attention/VIT_FSD50K/data_set/dev_22050_chunks"

folder_path3 = "/home/student.aau.dk/saales20/local-global-Mul-Head-Attention/VIT_FSD50K/data_set/eval_22050_chunks"



# List all files in the folder
files1 = os.listdir(folder_path)
files2 = os.listdir(folder_path1)
files3 = os.listdir(folder_path2)
files4 = os.listdir(folder_path3)


# Get the number of files
num_files1 = len(files1)
num_files2 = len(files2)
num_files3 = len(files3)
num_files4 = len(files4)



print(f"Number of files in the dev: {num_files1}")
print(f"Number of files in the eval: {num_files2}")
print(f"Number of files in dev chunks: {num_files3}")
print(f"Number of files in eval chunks: {num_files4}")

