import codecs
import spacy

from docopt import docopt
from collections import defaultdict
from parser import Parser


def main():
    """
    Creates a "knowledge resource" from triplets file
    """

    # Get the arguments
    args = docopt("""Parse the Wikipedia dump and create a triplets file, each line is formatted as follows: X\t\Y\tpath

    Usage:
        parse_wikipedia.py <wiki_file> <vocabulary_file> <out_file>

        <wiki_file> = the Wikipedia dump file
        <vocabulary_file> = a file containing the words to include
        <out_file> = the output file
    """)

    wiki_file = args['<wiki_file>']
    vocabulary_file = args['<vocabulary_file>']
    out_file = args['<out_file>']

    parser = Parser()

    # Load the phrase pair files
    with codecs.open(vocabulary_file, 'r', 'utf-8') as f_in:
        vocabulary = set([line.strip() for line in f_in])

    with codecs.open(wiki_file, 'r', 'utf-8') as f_in:
        with codecs.open(out_file, 'w', 'utf-8') as f_out:

	    i = 0
            # Read the next paragraph
            for sent in f_in:
		i += 1
		if i % 10000 == 0:
		    print(i)

                # Skip empty lines
                sent = sent.strip()
                if len(sent) == 0:
                    continue

                dependency_paths = parser.parse_sent(sent, vocabulary)
                if len(dependency_paths) > 0:
                    for (x, y), paths in dependency_paths.iteritems():
                        for path in paths:
                            print >> f_out, '\t'.join([x, y, path])


if __name__ == '__main__':
    main()
