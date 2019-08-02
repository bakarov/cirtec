import re


class Preprocesser:
    def __init__(self):
        self.nltk_stopwords_en = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
                                  "you've", "you'll",
                                  "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
                                  'she', "she's",
                                  'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',
                                  'theirs',
                                  'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these',
                                  'those', 'am',
                                  'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having',
                                  'do', 'does',
                                  'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
                                  'while',
                                  'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
                                  'during',
                                  'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on',
                                  'off', 'over',
                                  'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
                                  'how', 'all',
                                  'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
                                  'not', 'only',
                                  'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don',
                                  "don't",
                                  'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren',
                                  "aren't",
                                  'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn',
                                  "hasn't",
                                  'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't",
                                  'needn',
                                  "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren',
                                  "weren't", 'won',
                                  "won't", 'wouldn', "wouldn't"]
        self.pattern = re.compile('[^a-zа-яA-ZА-Я.,!-; ]+')
        self.pattern_brackets = re.compile('[\(\[].*?[\)\]]')

    def preprocess(self, text_, pipeline, add_pos=False, punct_tag='PUNCT'):
        text = self.pattern.sub('', self.pattern_brackets.sub('', text_))
        text = text.replace('et al.', '')
        indent = 4
        word_id = 1
        lemma_id = 2
        pos_id = 3
        sentences  = []
        tokenized = []
        for par in pipeline.process(text).split('\n\n'):
            for parsed_word in par.split('\n')[indent:]:
                word = parsed_word.split('\t')[word_id].lower()
                lemma = parsed_word.split('\t')[lemma_id].lower()
                pos = parsed_word.split('\t')[pos_id]
                if pos == punct_tag:
                    continue
                if add_pos:
                    word = '{}_{}'.format(lemma, pos)
                if lemma not in self.nltk_stopwords_en:
                    tokenized.append(lemma)
            sentences.append(tokenized)
        return sentences
