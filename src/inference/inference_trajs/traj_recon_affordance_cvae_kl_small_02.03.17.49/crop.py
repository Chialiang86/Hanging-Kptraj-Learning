import cv2, os, glob
import io
import numpy as np

from tqdm import tqdm
from PIL import Image, ImageSequence
from matplotlib.widgets import RectangleSelector

class ImageCutter:
    def __init__(self, file):
        self.file = file
        self.img = Image.open(file)
        self.frames = [np.array(frame.copy().convert("RGB"))
                        for frame in ImageSequence.Iterator(self.img)]

        self.pos = np.array([0,0,0,0])

    # def crop(self):
    #     self.pos = self.pos.astype(int)
    #     self.cropped_imgs =  [frame[self.pos[1]:self.pos[3], self.pos[0]:self.pos[2]]
    #             for frame in self.frames]
    #     self.save()

    def save(self):
        out_path = f'{os.path.splitext(self.file)[0]}.jpg'
        Image.fromarray(np.uint8(self.frames[0])).save(out_path)
        # self.imgs_pil = [Image.fromarray(np.uint8(img))
        #                  for img in self.cropped_imgs]
        # self.imgs_pil[0].save(self.file,
        #              save_all=True,
        #              append_images=self.imgs_pil[1:],
        #              duration=16,
        #              loop=0)


def main():

    gif_paths = glob.glob('*/*.gif')
    gif_paths.sort()

    for gif_path in tqdm(gif_paths):
        gif = ImageCutter(gif_path)
        gif.save()

    return

if __name__=="__main__":
    main() 