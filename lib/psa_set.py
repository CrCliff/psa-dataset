import requests
import io
import cv2
import numpy as np
from .psa_resource import PsaResource
from .psa_card import PsaCard
from typing import List
from PIL import Image


class PsaSet(PsaResource):
    def __init__(self, href: str, *, has_gallery: bool = False):
        super().__init__(href)

        #        self.set_name = href.split('/')[4]
        self.has_gallery = has_gallery
        self.cards = []

    def get_cards(self):

        if not self.html:
            self._load_content()

        for row in self.html.find("tr"):
            cols = row.find("td")

            if cols:
                img_col = cols[1]
                spec_col = cols[2]
                cardnumb_col = cols[3]
                item_col = cols[4]
                grade_col = cols[5]
                #                pop_col = cols[6]
                comments_col = cols[8]

                imgs = self._img_src_from_col(img_col)

                for url, img in imgs:
                    card = PsaCard(
                        url=url,
                        shape=img.shape,
                        img=img,
                        spec_id=spec_col.text.strip(),
                        card_numb=cardnumb_col.text.strip(),
                        item=item_col.text.strip(),
                        grade=grade_col.text.strip(),
                        #                        pop=int(pop_col.text.strip()),
                        comments=comments_col.text.strip(),
                    )
                    self.cards.append(card)

    def _img_src_from_col(self, img_col) -> List[str]:
        def get_img(url):
            response = requests.get(url)
            bytes_im = io.BytesIO(response.content)
            return (
                url,
                cv2.cvtColor(np.array(Image.open(bytes_im)), cv2.COLOR_RGB2BGR),
            )

        return map(get_img, img_col.absolute_links)
