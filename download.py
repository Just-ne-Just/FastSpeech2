import gdown

url = "https://drive.usercontent.google.com/download?id=11DUv3Sw8zgcI7a_zLsq4Ar3Zm02PAxzi&export=download&authuser=2&confirm=t&uuid=bfddaf12-b802-4ca3-9ec7-5668d897c71a"
output = "model_best.pth"
gdown.download(url, output, quiet=False)
