from tqdm import tqdm
import zipfile
import gdown
import os

output_filename = "logo2k-dataset.zip"
if not os.path.exists(output_filename):
    print(">>> Downloading")
    gdown.download(
        url="https://drive.google.com/uc?export=download&id=1IFDF7gyjnnyrns4Fm-Ui8sMloBsNY1EO",
        output=output_filename,
        quiet=False,
    )

print(">>> Unzipping")
zf = zipfile.ZipFile(output_filename)
for file in tqdm(zf.infolist(), ascii=True):
    zf.extract(file)

os.remove(output_filename)
os.rename("Logo-2K+/", "logo2k-dataset/")
