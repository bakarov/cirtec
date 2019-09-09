import re
import json
import optparse
from collections import Counter, defaultdict
from operator import itemgetter
from gensim.models.phrases import Phrases, Phraser
from ufal.udpipe import Model, Pipeline
from os import path, makedirs
from langdetect import detect_langs
from typing import Pattern, Tuple, List, Dict

nltk_stopwords_en = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll",
                     "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's",
                     'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs',
                     'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am',
                     'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
                     'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while',
                     'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during',
                     'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over',
                     'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
                     'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
                     'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't",
                     'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't",
                     'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't",
                     'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn',
                     "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won',
                     "won't", 'wouldn', "wouldn't"]

nltk_stopwords_ru = ['и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как', 'а', 'то', 'все', 'она', 'так',
                     'его', 'но', 'да',
                     'ты', 'к', 'у', 'же', 'вы', 'за', 'бы', 'по', 'только', 'ее', 'мне', 'было', 'вот', 'от', 'меня',
                     'еще', 'нет', 'о',
                     'из', 'ему', 'теперь', 'когда', 'даже', 'ну', 'вдруг', 'ли', 'если', 'уже', 'или', 'ни', 'быть',
                     'был', 'него', 'до',
                     'вас', 'нибудь', 'опять', 'уж', 'вам', 'ведь', 'там', 'потом', 'себя', 'ничего', 'ей', 'может',
                     'они', 'тут', 'где',
                     'есть', 'надо', 'ней', 'для', 'мы', 'тебя', 'их', 'чем', 'была', 'сам', 'чтоб', 'без', 'будто',
                     'чего', 'раз', 'тоже',
                     'себе', 'под', 'будет', 'ж', 'тогда', 'кто', 'этот', 'того', 'потому', 'этого', 'какой', 'совсем',
                     'ним', 'здесь',
                     'этом', 'один', 'почти', 'мой', 'тем', 'чтобы', 'нее', 'сейчас', 'были', 'куда', 'зачем', 'всех',
                     'никогда', 'можно',
                     'при', 'наконец', 'два', 'об', 'другой', 'хоть', 'после', 'над', 'больше', 'тот', 'через', 'эти',
                     'нас', 'про',
                     'всего', 'них', 'какая', 'много', 'разве', 'три', 'эту', 'моя', 'впрочем', 'хорошо', 'свою',
                     'этой', 'перед', 'иногда',
                     'лучше', 'чуть', 'том', 'нельзя', 'такой', 'им', 'более', 'всегда', 'конечно', 'всю', 'между']


def preprocess(pipeline, sentence, lemmatize=True, add_pos=False, punct_tag='PUNCT', text_entry_char=4):
    indent = 4
    word_id = 1
    lemma_id = 2
    pos_id = 3
    tokenized_par = []
    for par in pipeline.process(sentence).split('\n\n'):
        for parsed_word in par.split('\n')[indent:]:
            word = parsed_word.split('\t')[word_id].lower()
            lemma = parsed_word.split('\t')[lemma_id]
            pos = parsed_word.split('\t')[pos_id]
            if pos == punct_tag:
                continue
            if lemmatize:
                if lemma not in nltk_stopwords_en and lemma not in nltk_stopwords_ru:
                    tokenized_par.append(lemma)
            else:
                if word not in nltk_stopwords_en and word not in nltk_stopwords_ru:
                    tokenized_par.append(word)
    return tokenized_par[text_entry_char:]


def select_lang_pipeline(sentence: str, en_pipeline: Pipeline, ru_pipeline: Pipeline,
                         language_pointer=None) -> Pipeline:
    if not language_pointer:
        lang = detect_langs(sentence)
        language = lang[0].lang
    else:
        language = language_pointer
    if language == 'en':
        pipeline = en_pipeline
    elif language == 'ru':
        pipeline = ru_pipeline
    else:
        return None
    return pipeline


def normalize_sentence(sentence: str, re_pattern_1: Pattern, re_pattern_2: Pattern) -> str:
    return re_pattern_1.sub('', re_pattern_2.sub('', sentence.replace('et al', '').replace('\xad ', '').lower()))


def load_data(filename: str, manual_language=None) -> Tuple[List, List, List]:
    with open(filename, 'r') as f:
        citations = f.read().split('\n')
    # Download models here: https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-2998#
    en_model = Model.load(path.join('.', 'udpipe', 'english-ewt-ud-2.4-190531.udpipe'))
    en_pipeline = Pipeline(en_model, 'tokenize', Pipeline.DEFAULT, Pipeline.DEFAULT, 'conllu')
    ru_model = Model.load(path.join('.', 'udpipe', 'russian-syntagrus-ud-2.4-190531.udpipe'))
    ru_pipeline = Pipeline(ru_model, 'tokenize', Pipeline.DEFAULT, Pipeline.DEFAULT, 'conllu')
    pattern = re.compile('[^a-zа-яA-ZА-Я ]+')
    pattern_brackets = re.compile('[\(\[].*?[\)\]]')
    citations_ids = []
    citations_texts = []
    citations_texts_lemma = []
    citation_id_position = 4
    for citation in citations:
        try:
            citation_id = '_'.join(citation.split()[:citation_id_position])
            citation_id = citation.split()[1]
        except IndexError:
            print('Citation ID was not recognized for {}'.format(citation))
            continue
        assert ':' in citation_id, 'Citation ID parsed incorrectly'
        normalized_citation = normalize_sentence(citation, pattern, pattern_brackets)
        pipeline = select_lang_pipeline(normalized_citation, en_pipeline, ru_pipeline, manual_language)
        if not pipeline:
            print('Language has not been detected for {}'.format(citation_id))
            continue
        citations_ids.append(citation_id)
        citations_texts.append(preprocess(pipeline, normalized_citation, lemmatize=False))
        citations_texts_lemma.append(preprocess(pipeline, normalized_citation, lemmatize=True))
    return citations_ids, citations_texts, citations_texts_lemma


def get_n_grams(tokens: List, n: int, min_count: int = 1, threshold: int = 1, delimiter=b'_') -> List:
    tokens_ = tokens
    n_grams = []
    for i in range(n):
        n_gram = Phrases(tokens_, min_count=min_count, delimiter=delimiter, threshold=threshold)
        n_gram_phraser = Phraser(n_gram)
        tokens__ = n_gram_phraser[tokens_]
        n_grams.append(tokens__)
        tokens_ = tokens__
    return n_grams


def get_result(n_grams: List, citations_ids: List, max_n: int, output_path: str, min_n: int=2):
    result = defaultdict(lambda: defaultdict(lambda: []))
    for n in range(min_n, max_n):
        result_key = '{}-grams'.format(n)
        n_grams_global_counts = dict(Counter(token for sentence in n_grams[n] for token in set(sentence)))
        sentences_ids = {}
        for word in n_grams_global_counts.keys():
            sentences_ids[word] = [sentence_id_ for sentence_id_, sentence_ in enumerate(n_grams[n]) if
                                   word in sentence_]
        for sentence_id, sentence in enumerate(n_grams[n]):
            n_grams_local_counts = dict(Counter(n_grams[n][sentence_id]))
            citations_id = citations_ids[sentence_id]
            for word in sentence:
                word_result = {}
                word_result['w'] = word
                word_result['n'] = n_grams_global_counts[word]
                word_result['t'] = n_grams_local_counts[word]
                word_result['c'] = itemgetter(*sentences_ids[word])(citations_ids)
                result[result_key][citations_id].append(word_result)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dict(result), f, ensure_ascii=False)


if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('-i', '--input',
                      help='Path to the input file', default='./data/citcon4bundles.txt')
    parser.add_option('-o', '--output',
                      help='Path to the output directory', default='data/')
    parser.add_option('-c', '--count',
                      help='Ignore all words and bigrams with total collected count lower than this value', default=1)
    parser.add_option('-t', '--threshold',
                      help='Represent a score threshold for forming the phrases (higher means fewer phrases).',
                      default=1)
    parser.add_option('-n', '--n',
                      help='Maximum n for formed n-grams', default=6)
    options, args = parser.parse_args()
    max_n = options.n - 1
    citations_ids, citations_texts, citations_texts_lemma = load_data(options.input)
    n_grams = get_n_grams(citations_texts, n=max_n, min_count=options.count, threshold=options.threshold)
    n_grams_lemma = get_n_grams(citations_texts_lemma, n=max_n, min_count=options.count, threshold=options.threshold)
    get_result(n_grams, citations_ids, max_n, path.join(options.output, 'result.json'))
    get_result(n_grams_lemma, citations_ids, max_n, path.join(options.output, 'result_lemma.json'))
