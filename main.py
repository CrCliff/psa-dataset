import sys
from requests_html import HTMLSession
from lib import Preprocessor, Scraper
from lib.io import FileReader, FileWriter

def _help():
    print('Please provide a single argument:')
    print('\tpr: Preprocess data from data/ dir to data/processed')
    print('\tsc: Scrape data into data/ dir')
    

if __name__=='__main__':
    if len(sys.argv) < 2:
        _help()
        exit()
        
    _, cmd = sys.argv
    
    if cmd == 'pr':
        # Preprocess data
        fr = FileReader()
        fw = FileWriter()

        try:
            pr = Preprocessor(fr, fw)
            pr.preprocess()
        finally:
            fw.close()
            fr.close()
    elif cmd == 'sc':
        # Scrape data
        session = HTMLSession()
        fw = FileWriter()
        
        try:
            scraper = Scraper(session, fw)
            scraper.scrape()
        finally:
            fw.close()
            session.close()
        pass
    else:
        _help()
