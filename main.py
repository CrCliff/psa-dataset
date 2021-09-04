import sys
import argparse
from requests_html import HTMLSession
from lib import Preprocessor, Scraper
from lib.io import FileReader, FileWriter, S3Reader, S3Writer


def _parser():
    parser = argparse.ArgumentParser()
    commands = ["pr", "sc"]
    commands_str = ", ".join(map(lambda c: '"' + c + '"', commands))
    parser.add_argument(
        "command",
        metavar="<command>",
        type=str,
        nargs=1,
        choices=["pr", "sc"],
        help=f"the script command. allowed values are {commands_str}",
    )
    parser.add_argument(
        "--fin",
        type=str,
        required=False,
        help='the input file for preprocessing, required with command "pr"',
    )
    parser.add_argument(
        "--fout",
        type=str,
        required=False,
        help='the output file for preprocessing, required with command "pr"',
    )

    return parser


def _help():
    print("Please provide a single argument:")
    print("\tpr: Preprocess data from data/ dir to data/processed")
    print("\tsc: Scrape data into data/ dir")


if __name__ == "__main__":
    parser = _parser()
    args = parser.parse_args()

    (mode,) = args.command

    if mode == "pr":
        if args.fin is None or args.fout is None:
            parser.error("pr requires --fin and --fout")

        fin: str = args.fin
        fout: str = args.fout

        fr = FileReader()
        fw = FileWriter()
        s3r = S3Reader()
        s3w = S3Writer()

        try:
            pr = Preprocessor(fr, fw, s3r, s3w)

            # TODO: S3 -> local file / local file -> S3
            if fin.startswith("s3://") and fout.startswith("s3://"):
                # S3 to S3
                pr.preprocess_s3(s3_in=fin, s3_out=fout)
            else:
                # Local to Local
                pr.preprocess(file_in=fin, file_out=fout)
        finally:
            fw.close()
            fr.close()
    elif mode == "sc":
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
