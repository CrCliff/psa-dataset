import requests
import re
import time
import itertools
from bs4 import BeautifulSoup
from lib import PsaResource, PsaSet
from lib.io import FileWriter, FileReader

def batch(lst, n):
    iterator = iter(lst)

    for item in iterator:
        yield itertools.chain([item],
                itertools.islice(iterator, n-1))

def is_baseball_endpoint(href):
    PATTERN = '/psasetregistry/baseball/.*/\d+'
    p = re.compile(PATTERN)

    return p.match(href)

def endpoint(href):
    BASE_URL = 'https://www.psacard.com'

    return BASE_URL + href


def parse_set(href):
    psa_set = PsaSet(href, has_gallery=True)
    print(psa_set)

    try:
        psa_set.get_cards()
        time.sleep(0.75)
        yield from psa_set.cards
    except:
        print(f'ERROR: Failed to get cards for set {psa_set}')
        print(err)


def parse_set_group(href):
    url = endpoint(href)

    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')

    for link in soup.find_all('a'):
        href = link.get('href')
        if 'alltimefinest' in href:
            url = endpoint(href)
            page = requests.get(url)
            soup = BeautifulSoup(page.content, 'html.parser')

    set_links = []

    for row in soup.find_all('tr'):
        links = [l.get('href') for l in row.findChildren('a')]
        has_gallery = any('imagegallery' in l for l in links)
        if has_gallery:
            set_links.append(links[0])

    for href in set_links:
        yield from parse_set(href)

def parse_category(href):
    url = endpoint(href)

    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')

    for link in soup.find_all('a'):
        href = link.get('href')
        if is_baseball_endpoint(href):
#            print(href)
            yield from parse_set_group(href)

def get_cards():
    url = endpoint('/psasetregistry/baseball/1')

    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')

    pattern = '/psasetregistry/baseball/.*/\d+'
    p = re.compile(pattern)

    for link in soup.find_all('a'):
        href = link.get('href')
        if is_baseball_endpoint(href):
            yield from parse_category(href)

def get_cards_batched(n=100):
    cards = get_cards()
    yield from batch(get_cards(), n)

def main():
    file_writer = FileWriter()
    cards = get_cards_batched()

    i = 0
    for batch in cards:
        file_name = f'data/{i:04}.csv'

        file_writer.open(file_name)
        for card in batch:
            file_writer.write(card)
            file_writer.write('\n')
        file_writer.close()

        i += 1

if __name__=='__main__':
    main()
