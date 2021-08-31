import time
from typing import Generator, Iterable

import requests
from bs4 import BeautifulSoup
from requests_html import HTMLSession

from lib import PsaResource, PsaSet
from lib.io import FileWriter
from lib.iter import batch
from lib.psa_card import PsaCard

CardGenerator = Generator[PsaCard, None, None]

session = HTMLSession()


def endpoint(href):
    BASE_URL = "https://www.psacard.com"

    return BASE_URL + href


def parse_set(href) -> CardGenerator:
    psa_set = PsaSet(session, href)
    print(psa_set)

    try:
        time.sleep(0.75)
        yield from psa_set.get_cards()
    except Exception as err:
        print(f"ERROR: Failed to get cards for set {psa_set}")
        print(err)


def parse_set_group(href) -> CardGenerator:
    url = endpoint(href)
    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser")

    for link in soup.find_all("a"):
        href = link.get("href")
        if "alltimefinest" in href:
            url = endpoint(href)
            page = requests.get(url)
            soup = BeautifulSoup(page.content, "html.parser")

    set_links = []

    for row in soup.find_all("tr"):
        links = [l.get("href") for l in row.findChildren("a")]
        has_gallery = any("imagegallery" in l for l in links)
        if has_gallery:
            set_links.append(links[0])

    for href in set_links:
        yield from parse_set(href)


def parse_category(href) -> CardGenerator:
    url = endpoint(href)
    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser")

    for link in soup.find_all("a"):
        href = link.get("href")
        if PsaResource.is_setlist_endpoint(href):
            yield from parse_set_group(href)


def get_cards() -> CardGenerator:
    url = endpoint("/psasetregistry/baseball/1")

    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser")

    for link in soup.find_all("a"):
        href = link.get("href")
        if PsaResource.is_setlist_endpoint(href):
            yield from parse_category(href)


def get_cards_batched(n=100) -> Generator[Iterable[PsaCard], None, None]:
    yield from batch(get_cards(), n)


def main():
    file_writer = FileWriter()
    cards = get_cards_batched()

    i = 0
    for batch in cards:
        file_name = f"data/{i:04}.csv"

        file_writer.open(file_name)
        for card in batch:
            file_writer.write(card)
            file_writer.write("\n")
        file_writer.close()

        i += 1

    session.close()


if __name__ == "__main__":
    main()
