import re
import json
import optparse
from collections import Counter, defaultdict
from ufal.udpipe import Model, Pipeline
from os import path, makedirs, listdir
from langdetect import detect_langs
from typing import Pattern, Tuple, List, Dict
from nltk import ngrams


def preprocess(pipeline, sentence: str, stopwords: List, lemmatize: bool = True, punct_tag: str = 'PUNCT',
               text_entry_char: int = 4):
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
            if pos == punct_tag or word in stopwords:
                continue
            if lemmatize:
                tokenized_par.append(lemma)
            else:
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


def load_data(filename: str, lemmatize: bool, stopwords_path: str, manual_language=None) -> Tuple[List, List]:
    with open(filename, 'r') as f:
        citations = f.read().split('\n')
    stopwords = []
    for stopwords_list in listdir(stopwords_path):
        with open(path.join(stopwords_path, stopwords_list), 'r') as f:
            stopwords.extend(f.read().split('\n'))
    # Download models here: https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-2998#
    en_model = Model.load(path.join('.', 'udpipe', 'english-ewt-ud-2.4-190531.udpipe'))
    en_pipeline = Pipeline(en_model, 'tokenize', Pipeline.DEFAULT, Pipeline.DEFAULT, 'conllu')
    ru_model = Model.load(path.join('.', 'udpipe', 'russian-syntagrus-ud-2.4-190531.udpipe'))
    ru_pipeline = Pipeline(ru_model, 'tokenize', Pipeline.DEFAULT, Pipeline.DEFAULT, 'conllu')
    pattern = re.compile('[^a-zа-яA-ZА-Я ]+')
    pattern_brackets = re.compile('[\(\[].*?[\)\]]')
    citations_ids = []
    citations_texts = []
    citation_id_position = 4
    for citation in citations:
        try:
            citation_id = '_'.join(citation.split()[:citation_id_position])
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
        citations_texts.append(preprocess(pipeline, normalized_citation, stopwords, lemmatize=lemmatize))
    return citations_ids, citations_texts


def get_n_grams_local_counts(citations_texts: List, n: int, delimiter=' ') -> List:
    n_grams_with_counts = []
    for citations_text in citations_texts:
        n_grams = [delimiter.join(n_gram_tuples) for n_gram_tuples in list(ngrams(citations_text, n))]
        n_grams_counts = dict(Counter(n_grams))
        n_grams_with_counts.append(n_grams_counts)
    return n_grams_with_counts


def get_n_grams_global_counts(n_grams: List, threshold: int) -> Dict:
    all_global_counts = defaultdict(lambda: {'count': 0, 'ids': []})
    threshold_global_counts = defaultdict(lambda: {})
    for sentence_id, sentence in enumerate(n_grams):
        for n_gram, n_gram_count in sentence.items():
            if n_gram not in threshold_global_counts:
                current_count = all_global_counts[n_gram]['count'] + n_gram_count
                all_global_counts[n_gram]['ids'].append({citations_ids[sentence_id]: n_gram_count})
                if current_count >= threshold:
                    threshold_global_counts[n_gram]['count'] = current_count
                    threshold_global_counts[n_gram]['ids'] = all_global_counts[n_gram]['ids']
                else:
                    all_global_counts[n_gram]['count'] = current_count
            else:
                threshold_global_counts[n_gram]['count'] += n_gram_count
                threshold_global_counts[n_gram]['ids'].append({citations_ids[sentence_id]: n_gram_count})
    return dict(threshold_global_counts)


def serialize_n_grams_counts(n_grams_counts: Dict, n: int, output_dir):
    try:
        with open(path.join(output_dir, '{}-gram-result.json'.format(n)), 'w', encoding='utf-8') as f:
            json.dump(n_grams_counts, f, ensure_ascii=False)
    except FileNotFoundError:
        makedirs(output_dir)


if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('-i', '--input',
                      help='Path to the input file', default='./data/citcon4bundles.txt')
    parser.add_option('-o', '--output',
                      help='Path to the output directory', default='result')
    parser.add_option('-t', '--threshold',
                      help='Minimum global count for n-grams.', default=5)
    parser.add_option('-n', '--max_n',
                      help='Maximum n for formed n-grams', default=3)
    parser.add_option('-m', '--min_n',
                      help='Minimum n for formed n-grams', default=2)
    parser.add_option('-l', '--lemmatization',
                      help='Enable lemmatization for texts', default=True)
    parser.add_option('-s', '--stopwords_path',
                      help='Path to stop words lists', default='utils')
    options, args = parser.parse_args()
    citations_ids, citations_texts = load_data(options.input, options.lemmatization, options.stopwords_path)
    for n in range(options.min_n, options.max_n + 1):
        n_grams_counts = get_n_grams_local_counts(citations_texts, n)
        global_n_grams_counts = get_n_grams_global_counts(n_grams_counts, options.threshold)
        serialize_n_grams_counts(global_n_grams_counts, n, options.output)
