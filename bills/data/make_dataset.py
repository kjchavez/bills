from __future__ import print_function
import argparse
import json
import glob
import logging
import os
import random
import shutil
import subprocess

import schema

DEFAULT_DEV_SIZE = 3000
DEFAULT_TEST_SIZE = 3000

def bill_iterator(root, congress_num="*", bill_type="*", bill_id="*", version_type="*"):
    """ Yields bill directories for all congress data at |root|.

    The directory structure obtained by download.sh is:

        {congress_num}/bills/{type}/{bill id}/
          ...text_versions/{version type}/document.txt
          ...data.json

    """
    congress_num = str(congress_num)
    for billdir in glob.iglob(os.path.join(root, congress_num, "bills", bill_type, bill_id,
                                           "text-versions", version_type)):
        yield billdir

def read_data(billdir):
    """ Reads data from leaf bill directory. """
    with open(os.path.join(billdir, "data.json")) as fp:
        data = json.load(fp)

    with open(os.path.join(billdir, "document.txt")) as fp:
        text = fp.read()

    with open(os.path.join(billdir, os.pardir, os.pardir, "data.json")) as fp:
        metadata = json.load(fp)

    data['document'] = text
    data.update(metadata)
    return data

def clean_text(text):
    return text.lower()

def _subdict(full, keys):
    d = {}
    for k in keys:
        d[k] = full.get(k, None)
    return d

def get_label(x):
    """ Returns a dictionary of relevant target variables for an example |x|. """
    label = _subdict(x, ("status",))
    return label


def split_data(data, dev_size=None, test_size=None):
    """ Creates reasonable train/dev/test splits.

    Returns:
        (train, dev, test) splits in that order.
    """
    N = len(data)
    if test_size is None:
        test_size = min(DEFAULT_TEST_SIZE, N // 5)

    if dev_size is None:
        dev_size = min(DEFAULT_DEV_SIZE, (N - test_size) // 5)


    logging.info("N_dev=%d, N_test=%d", dev_size, test_size)
    if N < (dev_size + test_size)*2:
        raise ValueError("Not enough data for target dev/test size.")

    random.shuffle(data)
    dev = data[0:dev_size]
    test = data[dev_size:(dev_size+test_size)]
    train = data[(dev_size+test_size):]
    return train, dev, test

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrites existing files.")
    parser.add_argument("--max", type=int, default=-1,
                        help="Max number of examples.")
    return parser.parse_args()

def main():
    args = parse_args()

    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    raw_data_dir = os.path.join(project_dir, "data/raw")
    processed_data_dir = os.path.join(project_dir, "data/processed")

    examples = []
    for i, billdir in enumerate(bill_iterator(raw_data_dir, bill_type="s")):
        x = read_data(billdir)
        if i == 0:
            logging.info("Saving schemata of first example to: %s", "schemata.txt")
            with open(os.path.join("schemata.txt"), 'w') as fp:
                schema.print_schema(x, file=fp)

        # Create just processed data directory.
        example_dir = os.path.join(processed_data_dir, x["bill_version_id"])
        if not args.overwrite and os.path.exists(example_dir):
            logging.debug("Example %s already exists. Skipping.", x["bill_version_id"])
            continue

        os.makedirs(example_dir, exist_ok=True)

        # For now, we'll just create a symlink, but we might want to actually preprocess and write
        # new files.
        # Note: relative symbol link should be with respect to where the link lives
        subprocess.call(["ln", "-sf", os.path.abspath(os.path.join(billdir, "document.txt")),
                         os.path.join(example_dir, "document.txt")])

        with open(os.path.join(example_dir, "label.json"), 'w') as fp:
            label = get_label(x)
            json.dump(label, fp)

        examples.append(x["bill_version_id"])

        if (i+1) % 1000 == 0:
            logging.info("Processed %d examples.", i + 1)

        if (i+1) == args.max:
            logging.info("Hit max %d examples.", args.max)
            break


    print(examples)
    # Split into train, dev, test sets.
    train, dev, test = split_data(examples)
    for filename, split in [("train.txt", train), ("dev.txt", dev), ("test.txt", test)]:
        with open(os.path.join(processed_data_dir, filename), 'w') as fp:
            print('\n'.join(split), file=fp)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
