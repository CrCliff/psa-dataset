import sys
import numpy as np
import numpy.typing as npt
from typing import List


class PsaCard:
    def __init__(
        self,
        *,
        url: str,
        shape,
        img: npt.NDArray,
        spec_id,
        card_numb,
        item: str,
        grade: str,
        #                pop,
        comments: str
    ):
        self.url = url
        self.shape = shape
        self.img = img
        self.spec_id = spec_id
        self.card_numb = card_numb
        self.item = item
        self.grade = grade
        #        self.pop = pop
        self.comments = comments
        
    @staticmethod
    def from_csv(csv: str) -> 'PsaCard':
        cols = csv.split(',')
        
        url, shapes, img_flat, spec_id, card_numb, item, grade, comments = cols
        
        shape = tuple(map(int, shapes.split(' ')))
        img_splt = list(img_flat.split(' '))
        img = np.float32(np.array(img_splt).reshape(shape))

        return PsaCard(
            url=url,
            shape=shape,
            img=img,
            spec_id=spec_id,
            card_numb=card_numb,
            item=item,
            grade=grade,
            comments=comments,
        )

    def keys(self) -> List[str]:
        return [
            "url",
            "shape",
            "img",
            "spec_id",
            "card_numb",
            "item",
            "grade",
            #                'pop',
            "comments",
        ]

    def __str__(self) -> str:
        #        np.set_printoptions(threshold=sys.maxsize)
        return ",".join(
            [
                self.url,
                " ".join(map(str, self.shape)),
                " ".join(self.img.flatten().astype("str")),
                self.spec_id,
                self.card_numb,
                self.item,
                self.grade,
                #                self.pop,
                self.comments,
            ]
        )
