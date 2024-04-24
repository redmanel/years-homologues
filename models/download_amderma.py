import urllib.request
from tqdm import tqdm
import py7zr



class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

download_url("https://drive.usercontent.google.com/download?id=1yId9WdJ-06jK42kBa4T8SbJCZpdEHXo7&export=download&authuser=0&confirm=t",
            'Amderma.7z')

print('start extract')
with py7zr.SevenZipFile("Amderma.7z", 'r') as archive:
    archive.extractall(path="")
print('end extract')