import cv2
import itertools
import numpy as np
from PIL import Image
from .psa_card import PsaCard
from .image import resize_and_fill
from .io import FileReader, FileWriter

RATIO = 6/4
SIZE = (600,400)
 
# See https://www.psacard.com/resources/gradingstandards
QUALIFIERS = [
    'MK',
    'MC',
    'OC',
    'ST',
    'PD',
    'OF',
]

class Preprocessor:
   
    def __init__(self, file_reader: FileReader, file_writer: FileWriter, start: int=0):
        self.file_reader = file_reader
        self.file_writer = file_writer
        self.start = start
    
    def preprocess(self):
        fr = self.file_reader
        fw = self.file_writer
        
        j = 0
        for i in itertools.count(start=self.start):
            file_in = f"data/{i:04}.csv"
            file_out = f"data/processed/{i:04}.csv"
                    
            print(file_in, '->', file_out)
            
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
                        # print(err)
                        pass
                
            except Exception as err:
                print(err)
                break
            finally:
                fr.close()
                fw.close()
    
    @staticmethod
    def write_card(fw: FileWriter, card: PsaCard):
         if card.img.shape[0] > card.img.shape[1]:
            Preprocessor.grayscale(card)
            Preprocessor.resize(card)
            features = Preprocessor.featurize(card)

            sample = [Preprocessor.fgrade(card.grade), *features]
            sample_str = ','.join(map(str, sample))

            fw.write(sample_str)
            fw.write('\n')       

    @staticmethod
    def resize(card):
        x, y = card.img.shape
        
        new_shape = x,y
        if (x / y) < RATIO:
            new_x = int(x * RATIO)
            new_y = y
        else:
            new_x = x
            new_y = int(x / RATIO)        
        new_shape = new_x, new_y

        img = cv2.cvtColor(card.img, cv2.COLOR_GRAY2RGB)
        img = resize_and_fill(Image.fromarray(img, 'RGB'), new_shape[::-1])
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
            g = g.replace(qual, '')
        return float(g)

    
