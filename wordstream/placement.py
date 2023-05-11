import numpy
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from wordstream.draw import WordPlacement
from wordstream.util import Word


class Placement:
    def __init__(self, width: int, height: int, min_font_size: float, max_font_size: float, font_path: str):
        self.width = width
        self.height = height
        self.min_font_size = min_font_size
        self.max_font_size = max_font_size

        self.img = Image.new("L", (width, height))
        self.draw = ImageDraw.Draw(self.img)
        self.font_path = font_path

    def _word_coord_to_pixel_coord(self, word: WordPlacement) -> tuple[int, int]:
        # TODO: convert coordinates so that image center is y=0 and height is inverted etc.
        return round(word.x), round(word.y)

    def _word_box_to_pixel_box(self, word: WordPlacement) -> tuple[int, int, int, int]:
        # TODO: convert coordinates so that image center is y=0 and height is inverted etc.
        return round(word.x), round(word.y), round(word.x + word.width), round(word.y + word.height)

    def frequency_to_font_size(self, word: WordPlacement) -> int:
        # TODO: implement
        return 20

    def get_font(self, word: WordPlacement) -> ImageFont:
        return ImageFont.truetype(self.font_path, size=self.frequency_to_font_size(word))

    def get_size(self, word: WordPlacement) -> WordPlacement:
        font = self.get_font(word)
        box_size = self.draw.textbbox((0, 0), word.word.text, font=font, anchor="lt")
        word.width = box_size[2]
        word.height = box_size[3]
        return word

    def get_sprite(self, word: WordPlacement) -> WordPlacement:
        test_img = Image.new("L", (word.width, word.height))  # create image the size of the words bounding box
        draw = ImageDraw.Draw(test_img)
        draw.text((0, 0), word.word.text, fill=255, font=self.get_font(word), align="left", anchor="lt")

        word.sprite = numpy.asarray(test_img)
        return word

    def place(self, word: WordPlacement):
        self.draw.text(self._word_coord_to_pixel_coord(word), word.word.text, fill=255, font=self.get_font(word), align="left", anchor="lt")

    def check_placement(self, word: WordPlacement) -> bool:
        img_crop = self.img.crop(self._word_box_to_pixel_box(word))
        img_array = np.asarray(img_crop)
        assert img_array.shape == word.sprite.shape
        # convert to boolean array because uint8 will overflow on multiplication
        overlap = (img_array > 0) * (word.sprite > 0)
        return overlap.sum() == 0


if __name__ == '__main__':
    word = WordPlacement(x=0, y=0, height=0, width=0, word=Word("test", frequency=3, sudden=3), sprite=None)
    placement = Placement(1000, 1000, min_font_size=10, max_font_size=30, font_path="../fonts/RobotoMono-VariableFont_wght.ttf")

    placement.get_size(word)
    placement.get_sprite(word)

    print(placement.check_placement(word))
    placement.place(word)
    word.x = 10
    print(placement.check_placement(word))
