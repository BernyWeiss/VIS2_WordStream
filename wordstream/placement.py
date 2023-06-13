from typing import Callable
from dataclasses import dataclass
import numpy
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from wordstream.util import Word


def _value_map(from_start, from_stop, to_start, to_stop, round_value=False) -> Callable:
    map_func = lambda v: (v - from_start) / (from_stop - from_start) * (to_stop - to_start) + to_start
    if round_value:
        return lambda v: round(map_func(v))
    else:
        return map_func


@dataclass
class WordPlacement:
    """Class to store information about a placement of a word in the figure"""
    x: float
    """x coordinate of position"""
    y: float
    """y coordinate of position"""
    width: float
    """width of bounding box of word"""
    height: float
    """height of bounding box of word"""
    font_size: int  # in pt
    """Size of font in pt"""
    word: Word
    """Word which should be placed"""
    sprite: np.ndarray | None = None
    """Pixel-sprite of word (depends on fontsize)"""
    placed: bool = False
    """Whether or not the word has been placed in figure"""

    def to_dict(self) -> dict:
        """Transforms WordPlacement to dict of position (x and y), fontsize and text"""
        return {"x": self.x, "y": self.y, "font_size": self.font_size, "text": self.word.text}


class Placement:
    """Figure where placement of words can be calculated"""
    def __init__(self, width: int, height: int, ppi: int, max_sudden: float, min_font_size: float, max_font_size: float, font_path: str):
        # interpret width and height as inches and calculate pixel width and height from ppi
        #  + font size from pt to px
        self.ppi = ppi
        """Resolution points per inch for placement calculation"""
        self.width = width
        """width of figure in inches"""
        self.height = height
        """height of figure in inches"""

        self.width_map = _value_map(0, width, 0, width * ppi, round_value=True)
        self.width_px = self.width_map(width)
        """width of figure in pixel"""

        # since original height is centered around 0 we map from -height/2 and height/2
        self.height_map = _value_map(-height / 2, height / 2, 0, height * ppi, round_value=True)

        # since draw.textbbox returns height and width in px we have to convert back to inches
        self.box_width_map = self.width_map
        self.inv_box_width_map = _value_map(0, width * ppi, 0, width, round_value=False)

        self.box_height_map = _value_map(0, height, 0, height * ppi, round_value=True)
        self.inv_box_height_map = _value_map(0, height * ppi, 0, height, round_value=False)
        self.height_px = self.box_height_map(height)
        """height of figure in pixel"""

        self.font_map = _value_map(0, max_sudden, self.pt_to_px(min_font_size), self.pt_to_px(max_font_size), round_value=True)
        """Map to calculate fort size based on sudden attention measure"""
        self.img = Image.new("L", (self.width_px, self.height_px))
        """Image used to calculate placements and check collisions"""
        self.draw = ImageDraw.Draw(self.img)
        self.font_path = font_path
        """Path where font is stored"""

    def _word_coord_to_pixel_coord(self, word: WordPlacement) -> tuple[int, int]:
        return self.width_map(word.x), self.height_map(word.y)

    def _word_box_to_pixel_box(self, word: WordPlacement) -> tuple[int, int, int, int]:
        return self.width_map(word.x), self.height_map(word.y), self.width_map(word.x + word.width), self.height_map(word.y + word.height)

    def pt_to_px(self, pt: float) -> float:
        """calculates pixel from pt"""
        # 1 pt = 1/72 inches
        inches = pt / 72
        return inches * self.ppi

    def px_to_pt(self, px: int) -> float:
        """calculates pt from pixel"""
        inches = px / self.ppi
        return inches * 72

    def get_font(self, word: WordPlacement) -> ImageFont:
        """Gets the font object used for a WordPlacement"""
        return ImageFont.truetype(self.font_path, size=self.font_map(word.word.sudden), layout_engine=ImageFont.LAYOUT_BASIC)

    def get_size(self, word: WordPlacement) -> WordPlacement:
        """Sets the size of the bounding box for Word in WordPlacement"""
        font = self.get_font(word)
        box_size = self.draw.textbbox((0, 0), word.word.text, font=font, anchor="la")
        # since box size is in pixel we have to inverse map back to inches
        word.width = self.inv_box_width_map(box_size[2])
        word.height = self.inv_box_height_map(box_size[3])
        return word

    def get_sprite(self, word: WordPlacement) -> WordPlacement:
        """Sets the sprite of the wordplacement by creating an image the size of the words bounding box.
         bounding box is in pixel; to get inches must be converted back."""
        test_img = Image.new("L", (self.box_width_map(word.width), self.box_height_map(word.height)))
        draw = ImageDraw.Draw(test_img)
        draw.text((0, 0), word.word.text, font=self.get_font(word), fill=255, align="left", anchor="la")

        word.sprite = numpy.asarray(test_img)
        return word

    def place(self, word: WordPlacement):
        """places word on canvas"""
        word.placed = True
        word.font_size = self.px_to_pt(self.font_map(word.word.sudden))
        self.draw.text(self._word_coord_to_pixel_coord(word), word.word.text, fill=255, font=self.get_font(word), align="left", anchor="la")
        return (self, word)

    def check_placement(self, word: WordPlacement) -> bool:
        """Checks is word has overlap with already placed word in figure.

        returns true if there is no collision, else false.
        """
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
