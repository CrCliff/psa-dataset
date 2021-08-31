import re
import requests
from requests_html import HTMLSession
from bs4 import BeautifulSoup
from typing import Optional


class PsaResource:

    BASE_URL = "https://www.psacard.com"

    def __init__(self, href: str, base_url: Optional[str] = BASE_URL):

        if not isinstance(href, str):
            raise TypeError("href must be of type str")

        if not isinstance(base_url, str):
            raise TypeError("base_url must be of type str")

        self.href = href
        self.base_url = base_url
        self.html: Optional[BeautifulSoup] = None

    @staticmethod
    def is_set_endpoint(href: str) -> bool:

        if not isinstance(href, str):
            raise TypeError("href must be of type str")

        pattern = "/psasetregistry/baseball/.*/alltimeset/\d+"
        p = re.compile(pattern)

        return True if p.match(href) else False

    @staticmethod
    def is_setlist_endpoint(href: str) -> bool:

        if not isinstance(href, str):
            raise TypeError("href must be of type str")

        pattern = "/psasetregistry/baseball/.*/\d+"
        p = re.compile(pattern)

        return True if p.match(href) else False

    def _load_content(self) -> BeautifulSoup:
        if not self.html:
            url = self._endpoint()
            session = HTMLSession()

            page = session.get(url)
            page.html.render()
            self.html = page.html
            return self.html
            # self.html = BeautifulSoup(page.html, 'html.parser')

    def _endpoint(self) -> str:
        return self.base_url + self.href

    def __str__(self) -> str:
        return self._endpoint()
