import requests
import io
import cv2
import numpy as np
import numpy.typing as npt
from .psa_resource import PsaResource
from .psa_card import PsaCard
from typing import Generator, Iterator, List, Tuple
from PIL import Image

class PsaSet(PsaResource):
    def __init__(self, href: str):
        super().__init__(href)

    def get_cards(self) -> List[PsaCard]:

        html = self._load_content()

        cards: List[PsaCard] = []
        for row in html.find("tr"):
            cols = row.find("td")
            if cols:
                cards.extend(self._card_from_cols(cols))

        return cards

    def _card_from_cols(self, cols) -> Generator[PsaCard, None, None]:
        img_col = cols[1]
        spec_col = cols[2]
        cardnumb_col = cols[3]
        item_col = cols[4]
        grade_col = cols[5]
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
                comments=comments_col.text.strip(),
            )
            yield card

    def _img_src_from_col(self, img_col) -> Iterator[Tuple[str, npt.NDArray]]:
        def get_img(url):
            response = requests.get(url)
            bytes_im = io.BytesIO(response.content)
            return (
                url,
                cv2.cvtColor(np.array(Image.open(bytes_im)), cv2.COLOR_RGB2BGR),
            )

        return map(get_img, img_col.absolute_links)
