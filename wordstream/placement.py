from typing import Callable
from dataclasses import dataclass
import numpy
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from wordstream.util import Word


def value_map(from_start, from_stop, to_start, to_stop, round_value=False) -> Callable:
    map_func = lambda v: (v - from_start) / (from_stop - from_start) * (to_stop - to_start) + to_start
    if round_value:
        return lambda v: round(map_func(v))
    else:
        return map_func


@dataclass
class WordPlacement:
    x: float
    y: float
    width: float
    height: float
    font_size: int  # in pt
    word: Word
    sprite: np.ndarray | None = None
    placed: bool = False

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "font_size": self.font_size, "text": self.word.text}


class Placement:
    def __init__(self, width: int, height: int, ppi: int, max_sudden: float, min_font_size: float, max_font_size: float, font_path: str):
        # interpret width and height as inches and calculate pixel width and height from ppi
        #  + font size from pt to px
        self.ppi = ppi
        self.width = width
        self.height = height

        self.width_map = value_map(0, width, 0, width*ppi, round_value=True)
        self.width_px = self.width_map(width)

        # since original height is centered around 0 we map from -height/2 and height/2
        self.height_map = value_map(-height/2, height/2, 0, height*ppi, round_value=True)

        # since draw.textbbox returns height and width in px we have to convert back to inches
        self.box_width_map = self.width_map
        self.inv_box_width_map = value_map(0, width*ppi, 0, width, round_value=False)

        self.box_height_map = value_map(0, height, 0, height*ppi, round_value=True)
        self.inv_box_height_map = value_map(0, height*ppi, 0, height, round_value=False)
        self.height_px = self.box_height_map(height)

        self.font_map = value_map(0, max_sudden, self.pt_to_px(min_font_size), self.pt_to_px(max_font_size), round_value=True)

        self.img = Image.new("L", (self.width_px, self.height_px)) # why does this need 8bit? why not 1?
        self.draw = ImageDraw.Draw(self.img)
        self.font_path = font_path

    def _word_coord_to_pixel_coord(self, word: WordPlacement) -> tuple[int, int]:
        return self.width_map(word.x), self.height_map(word.y)

    def _word_box_to_pixel_box(self, word: WordPlacement) -> tuple[int, int, int, int]:
        return self.width_map(word.x), self.height_map(word.y), self.width_map(word.x + word.width), self.height_map(word.y + word.height)

    def pt_to_px(self, pt: float) -> float:
        # 1 pt = 1/72 inches
        inches = pt / 72
        return inches * self.ppi

    def px_to_pt(self, px: int) -> float:
        inches = px / self.ppi
        return inches * 72

    def get_font(self, word: WordPlacement) -> ImageFont:
        return ImageFont.truetype(self.font_path, size=self.font_map(word.word.sudden))

    def get_size(self, word: WordPlacement) -> WordPlacement:
        font = self.get_font(word)
        box_size = self.draw.textbbox((0, 0), word.word.text, font=font, anchor="lt")
        # since box size is in pixel we have to inverse map back to inches
        word.width = self.inv_box_width_map(box_size[2])
        word.height = self.inv_box_height_map(box_size[3])
        return word

    def get_sprite(self, word: WordPlacement) -> WordPlacement:
        # create image the size of the words bounding box -> convert back from inches to px for box size
        test_img = Image.new("L", (self.box_width_map(word.width), self.box_height_map(word.height)))
        draw = ImageDraw.Draw(test_img)
        draw.text((0, 0), word.word.text, fill=255, font=self.get_font(word), align="left", anchor="lt")

        word.sprite = numpy.asarray(test_img)
        return word

    def place(self, word: WordPlacement):
        word.placed = True
        word.font_size = self.px_to_pt(self.font_map(word.word.sudden))
        self.draw.text(self._word_coord_to_pixel_coord(word), word.word.text, fill=255, font=self.get_font(word), align="left", anchor="lt")
        return (self, word)

    def check_placement(self, word: WordPlacement) -> bool:
        img_crop = self.img.crop(self._word_box_to_pixel_box(word))
        img_array = np.asarray(img_crop)
        assert img_array.shape == word.sprite.shape
        # convert to boolean array because uint8 will overflow on multiplication
        overlap = (img_array > 0) * (word.sprite > 0)
        return overlap.sum() == 0


if __name__ == '__main__':
    word = WordPlacement(x=10, y=10, height=0, width=0, font_size=0, word=Word("test", frequency=3, sudden=3))
    placement = Placement(10, 4, ppi=100, min_font_size=10, max_sudden=10, max_font_size=30, font_path="../fonts/RobotoMono-VariableFont_wght.ttf")

    placement.get_size(word)
    placement.get_sprite(word)

    print(placement.check_placement(word))
    placement.place(word)
    print(np.asarray(placement.img).sum())
    word.x = 0.1
    print(placement.check_placement(word))
