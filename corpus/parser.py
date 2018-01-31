# -*- coding: utf-8 -*-
from mtoken import MToken
from collections import defaultdict
from nltk.parse.stanford import StanfordDependencyParser
from nltk.tokenize import StanfordSegmenter

class Parser():
    def __init__(self):
        path_to_jar = '/Users/semsevens/nlp/stanford-parser-full-2017-06-09/stanford-parser.jar'
        path_to_model_jar = '/Users/semsevens/nlp/stanford-parser-full-2017-06-09/stanford-parser-3.8.0-models.jar'
        model_path = '/Users/semsevens/nlp/stanford-chinese-corenlp-2017-06-09-models/edu/stanford/nlp/models/lexparser/chinesePCFG.ser.gz'
        self.parser = StanfordDependencyParser(path_to_jar, path_to_model_jar, model_path)

        self.segmenter = StanfordSegmenter(
            path_to_sihan_corpora_dict="/Users/semsevens/nlp/stanford-segmenter/data",
            path_to_model="/Users/semsevens/nlp/stanford-segmenter/data/pku.gz",
            path_to_dict="/Users/semsevens/nlp/stanford-segmenter/data/dict-chris6.ser.gz"
        )

        self.MAX_PATH_LEN = 4
        self.ROOT = 0
        self.UP = 1
        self.DOWN = 2
        self.SAT = 3
        #self.parse_tokens(sent, vocabulary)

    def init_tokens(self, sent):
        result = list(self.parser.parse(sent.split()))
        tokens = [MToken(None, None, None, None)]
        tokens = {}
        tokens['0'] = MToken(None, None, None, None)
        children = defaultdict(list)

        for token in result[0].to_conll(10).strip().split('\n'):
            print(token)
            token_list = token.split('\t')
            key = token_list[0]
            lemma = token_list[1]
            pos = token_list[3]
            head  = token_list[-4]
            dep = token_list[-3]
            tokens[key] = MToken(lemma, pos, head , dep)
            children[head].append(key)
        self.tokens = tokens
        self.children = children
        return tokens, children

    def parse_sent(self, sent, vocabulary=None):
        return self.parse_tokens(self.segmenter.segment(sent), vocabulary)

    def parse_tokens(self, sent, vocabulary=None):
        tokens, children = self.init_tokens(sent)
        if vocabulary:
            indices = [i for i, token in tokens.items() if token.dep and token.lemma in vocabulary]
        else:
            indices = [i for i, token in tokens.items() if token.dep]
        indices = sorted(indices, key=lambda x: int(x))

        pairs = []
        for i, index_i in enumerate(indices[:-1]):
            for index_j in indices[i+1:]:
                pairs.append((index_i, index_j))

        paths = defaultdict(list)
        [paths[pair].append(self.shortest_path(tokens, pair)) for pair in pairs]
        for pair, path in paths.items():
            print(pair, path)

        satellites = defaultdict(list)
        [satellites[pair].extend([sat_path for path in paths[pair] for sat_path in self.get_satellite_links(tokens, children, path)
                                    if sat_path is not None]) for pair in paths.keys()]

        for pair, path in satellites.items():
            print(pair, path)

        filtered_paths = defaultdict(list)
        [filtered_paths[(tokens[x].lemma, tokens[y].lemma)].extend(filter(None, [
            self.pretty_print(set_x_l, x, set_x_r, hx, lch, hy, set_y_l, y, set_y_r)
            for (set_x_l, x, set_x_r, hx, lch, hy, set_y_l, y, set_y_r) in satellites[(x, y)]]))
            for (x, y) in satellites.keys()]

        #for (x, y), paths in filtered_paths.items():
        #    for path in paths:
        #        print('\t'.join([tokens[x].lemma, tokens[y].lemma, path]))

        return filtered_paths

    def shortest_path(self, tokens, pair):

        # Get the root token and work on it
        if pair is None:
            return None

        x_token, y_token = pair

        # Get the path from the root to each of the tokens
        hx = self.heads(tokens, x_token)
        hy = self.heads(tokens, y_token)

        # 1. x is the head of y: "[parrot] and other [birds]"
        if hx == [] and x_token in hy:
            hy = hy[:hy.index(x_token)]
            hx = []
            lch = x_token

        # 2. y is the head of x: "[birds] such as [parrots]"
        elif hy == [] and y_token in hx:
            hx = hx[:hx.index(y_token)]
            hy = []
            lch = y_token

        elif len(hx) == 0 or len(hy) == 0:
            return None

        # 3. x and y have no common head - the first head in each list should be the sentence root, so
        # this is possibly a parse error?
        elif hy[0] != hx[0]:
            return None

        # 4. x and y are connected via a direct parent or have the exact same path to the root, as in "[parrot] is a [bird]"
        elif hx == hy:
            lch = hx[-1]
            hx = hy = []

        # 5. x and y have a different parent which is non-direct, as in "[parrot] is a member of the [bird] family".
        # The head is the last item in the common sequence of both head lists.
        else:
            for i in xrange(min(len(hx), len(hy))):
                # Now we've found the common ancestor in i-1
                if hx[i] is not hy[i]:
                    break

            if len(hx) > i:
                lch = hx[i-1]
            elif len(hy) > i:
                lch = hy[i-1]
            else:
                return None

            # The path from x to the lowest common head
            hx = hx[i+1:]

            # The path from the lowest common head to y
            hy = hy[i+1:]

        hx = hx[::-1]

        return (x_token, hx, lch, hy, y_token)

    def heads(self, tokens, index):
        hs = []
        while tokens[index].head != '0':
            index = tokens[index].head
            hs.append(index)
        return hs[::-1]


    def get_satellite_links(self, tokens, children, path):
        if path is None:
            return []

        x_tokens, hx, lch, hy, y_tokens = path
        paths = [(None, x_tokens, None, hx, lch, hy, None, y_tokens, None)]
        tokens_on_path = set([x_tokens] + hx + [lch] + hy + [y_tokens])

        # Get daughters of x not in the path
        set_xs = [child for child in children[x_tokens] if child not in tokens_on_path]
        set_ys = [child for child in children[y_tokens] if child not in tokens_on_path]

        for child in set_xs:
            if int(child) < int(x_tokens):
                paths.append((child, x_tokens, None, hx, lch, hy, None, y_tokens, None))
            else:
                paths.append((None, x_tokens, child, hx, lch, hy, None, y_tokens, None))

        for child in set_ys:
            if int(child) < int(x_tokens):
                paths.append((None, x_tokens, None, hx, lch, hy, child, y_tokens, None))
            else:
                paths.append((None, x_tokens, None, hx, lch, hy, None, y_tokens, child))

        return paths

    def pretty_print(self, set_x_l, x, set_x_r, hx, lch, hy, set_y_l, y, set_y_r):
        """
        Filter out long paths and pretty print the short ones
        :return: the string representation of the path
        """
        set_path_x_l = []
        set_path_x_r = []
        set_path_y_r = []
        set_path_y_l = []
        lch_lst = []

        if set_x_l:
            set_path_x_l = [self.edge_to_string(set_x_l) + '/' + self.direction(self.SAT)]
        if set_x_r:
            set_path_x_r = [self.edge_to_string(set_x_r) + '/' + self.direction(self.SAT)]
        if set_y_l:
            set_path_y_l = [self.edge_to_string(set_y_l) + '/' + self.direction(self.SAT)]
        if set_y_r:
            set_path_y_r = [self.edge_to_string(set_y_r) + '/' + self.direction(self.SAT)]

        # X is the head
        if lch == x:
            dir_x = self.direction(self.ROOT)
            dir_y = self.direction(self.DOWN)
        # Y is the head
        elif lch == y:
            dir_x = self.direction(self.UP)
            dir_y = self.direction(self.ROOT)
        # X and Y are not heads
        else:
            lch_lst = [self.edge_to_string(lch, is_head=True) + '/' + self.direction(self.ROOT)] if lch else []
            dir_x = self.direction(self.UP)
            dir_y = self.direction(self.DOWN)

        len_path = len(hx) + len(hy) + len(set_path_x_r) + len(set_path_x_l) + \
                   len(set_path_y_r) + len(set_path_y_l) + len(lch_lst)

        if len_path <= self.MAX_PATH_LEN:
            cleaned_path = '_'.join(set_path_x_l + [self.argument_to_string(x, 'X') + '/' + dir_x] + set_path_x_r +
                                    [self.edge_to_string(token) + '/' + self.direction(self.UP) for token in hx] +
                                    lch_lst +
                                    [self.edge_to_string(token) + '/' + self.direction(self.DOWN) for token in hy] +
                                    set_path_y_l + [self.argument_to_string(y, 'Y') + '/' + dir_y] + set_path_y_r)
            return cleaned_path
        else:
            return None

    def edge_to_string(self, t, is_head=False):
        """
        Converts the token to an edge string representation
        :param token: the token
        :return: the edge string
        """
        t = self.tokens[t]
        #return '/'.join([t.lemma.strip().lower(), t.pos, t.dep if t.dep != '' and not is_head else 'ROOT'])
        return '/'.join([t.lemma.strip().lower(), t.pos, t.dep])


    def argument_to_string(self, token, edge_name):
        """
        Converts the argument token (X or Y) to an edge string representation
        :param token: the X or Y token
        :param edge_name: 'X' or 'Y'
        :return:
        """
        token = self.tokens[token]
        return '/'.join([edge_name, token.pos, token.dep])


    def direction(self, dir):
        """
        Print the direction of the edge
        :param dir: the direction
        :return: a string representation of the direction
        """
        # Up to the head
        if dir == self.UP:
            return '>'
        # Down from the head
        elif dir == self.DOWN:
            return '<'
        elif dir == self.SAT:
            return 'V'
        else:
            return '^'


if __name__ == '__main__':
    #s = u"你 有个 优惠券 快要 过期 了"
    #s = u"你 有 个 优惠券 快要 过期 了"
    #s = u"美团,是一家没有底线没有节操的公司。"
    vocabulary = set([u'美团', u'公司', u'节操'])
    parser = Parser()
    #sent = u"美团 , 是 一 家 没有 底线 没有 节操 的 公司 。"
    #parser.parse_tokens(sent, vocabulary)
    sent = u"美团,是一家没有底线没有节操的公司。"
    parser.parse_sent(sent, vocabulary)
