import os
## change working dir to the dir that contains this file
os.chdir(os.path.dirname(os.path.realpath(__file__))) 
image_files = []
os.chdir(os.path.join("data", "obj"))
for filename in os.listdir(os.getcwd()):
    if filename.endswith(".png"):
        image_files.append("/content/darknet/data/obj/" + filename)
os.chdir("..")
with open("train.txt", "w") as outfile:
    for image in image_files:
        outfile.write(image)
        outfile.write("\n")
    outfile.close()
os.chdir("..")