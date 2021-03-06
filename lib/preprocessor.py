from typing import Optional, Tuple
import cv2
import itertools
import numpy as np
from PIL import Image
from .psa_card import PsaCard
from .image import resize_and_fill
from .io import FileReader, FileWriter, S3Reader, S3Writer

RATIO = 6 / 4
SIZE = (600, 400)

# See https://www.psacard.com/resources/gradingstandards
QUALIFIERS = [
    "MK",
    "MC",
    "OC",
    "ST",
    "PD",
    "OF",
]

# TODO
TMP_S3_IN = '/tmp/s3.in'
TMP_S3_OUT = '/tmp/s3.out'

class Preprocessor:
    def __init__(
        self,
        file_reader: FileReader,
        file_writer: FileWriter,
        s3_reader: S3Reader,
        s3_writer: S3Writer,
    ):
        self.file_reader = file_reader
        self.file_writer = file_writer
        self.s3_reader = s3_reader
        self.s3_writer = s3_writer

    def _preprocess(self, file_in: str, file_out: str) -> None:
        fr = self.file_reader
        fw = self.file_writer

        try:
            fr.open(file_in)
            fw.open(file_out)

            for line in fr.read_lines():
                try:
                    card = PsaCard.from_csv(line)
                    Preprocessor.write_card(fw, card)
                except ValueError as err:
                    # In case of an issue converting the line to a card, just skip the line
                    # This can happen if the name of the card contains a comma
                    # Pobody's nerfect
                    print(err)
                    pass
        except Exception as err:
            print(err)
        finally:
            fr.close()
            fw.close()
    
    def preprocess_s3(self, *, s3_in: str, s3_out: str):
        # s3_in and s3_out should be in format "s3://my_bucket/some/key"
        # TODO: REGEX check

        # get bucket and key from full url
        in_splt = s3_in.split("/")
        in_bucket = in_splt[2]
        in_key = "/".join(in_splt[3:])

        out_splt = s3_out.split("/")
        out_bucket = out_splt[2]
        out_key = "/".join(out_splt[3:])

        print(s3_in, '->', s3_out)

        # copy file to tmp location, do preprocessing, and copy out file to bucket
        self.s3_reader.read(TMP_S3_IN, in_bucket, in_key)
        self._preprocess(TMP_S3_IN, TMP_S3_OUT)
        self.s3_writer.write(TMP_S3_OUT, out_bucket, out_key)

    def preprocess(self, *, file_in: str, file_out: str):
        self._preprocess(file_in, file_out)

    @staticmethod
    def write_card(fw: FileWriter, card: PsaCard):
        Preprocessor.grayscale(card)

        if card.img.shape[0] < card.img.shape[1]:
            card = Preprocessor.rotate(card)

        Preprocessor.resize(card)
        features = Preprocessor.featurize(card)

        sample = [Preprocessor.fgrade(card.grade), *features]
        sample_str = ",".join(map(str, sample))

        fw.write(sample_str)
        fw.write("\n")
    
    @staticmethod
    def rotate(card):
        img = cv2.rotate(card.img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        card.img = np.array(img)
        return card

    @staticmethod
    def resize(card):
        x, y = card.img.shape

        new_shape = x, y
        if (x / y) < RATIO:
            new_x = int(x * RATIO)
            new_y = y
        else:
            new_x = x
            new_y = int(x / RATIO)
        new_shape = new_x, new_y

        img = cv2.cvtColor(card.img, cv2.COLOR_GRAY2RGB)
        img = resize_and_fill(Image.fromarray(img, "RGB"), new_shape[::-1])
        img = cv2.resize(np.uint8(img), SIZE[::-1], interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(np.uint8(img), cv2.COLOR_RGB2GRAY)
        del card.img

        card.img = img
        return card

    @staticmethod
    def grayscale(card):
        img = np.uint8(card.img)

        # greyscale each image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        card.img = np.array(img)
        return card

    @staticmethod
    def featurize(card):
        # flatten each image
        return card.img.flatten()

    @staticmethod
    def fgrade(grade: str) -> float:
        g = grade
        for qual in [q for q in QUALIFIERS if q in grade]:
            g = g.replace(qual, "")
        return float(g)
