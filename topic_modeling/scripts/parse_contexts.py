from gensim.models import LdaMulticore
from gensim.corpora import Dictionary
from os import path, makedirs
from collections import Counter, defaultdict
from ufal.udpipe import Model, Pipeline
from langdetect import detect_langs
from json import dump, load
from sklearn.feature_extraction.text import TfidfVectorizer

# import pyLDAvis.gensim
import optparse
import re
import numpy as np
import warnings

warnings.filterwarnings('ignore')

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


def read_data(data_path=path.join('..', 'data', 'citcon4bundles.txt')):
    with open(data_path, 'r') as f:
        lines = f.read().split('\n')
    return lines


def select_lang_pipeline(sentence, en_pipeline, ru_pipeline):
    lang = detect_langs(sentence)
    language = lang[0].lang
    if language == 'en':
        pipeline = en_pipeline
    elif language == 'ru':
        pipeline = ru_pipeline
    else:
        raise ValueError('Unvalid language detected!')
    return pipeline


def preprocess(pipeline, sentence, add_pos=False, punct_tag='PUNCT'):
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
            if add_pos:
                word = '{}_{}'.format(lemma, pos)
            if lemma not in nltk_stopwords_en:
                tokenized_par.append(lemma)
    return ' '.join(tokenized_par)


def create_context_groups(lines):
    context_groups = defaultdict(lambda: {})
    errors = []
    pattern = re.compile('([^\s\w]|_)+')
    pattern_brackets = re.compile('[\(\[].*?[\)\]]')
    en_model = Model.load(path.join('.', '..', '..', 'udpipe', 'english-ewt-ud-2.3.udpipe'))
    en_pipeline = Pipeline(en_model, 'tokenize', Pipeline.DEFAULT, Pipeline.DEFAULT, 'conllu')
    ru_model = Model.load(path.join('.', '..', '..', 'udpipe', 'russian-syntagrus-ud-2.3.udpipe'))
    ru_pipeline = Pipeline(ru_model, 'tokenize', Pipeline.DEFAULT, Pipeline.DEFAULT, 'conllu')
    for line in lines:
        try:
            context_group, text = line.split(' ', 1)
            splits = text.split(' ', 3)
            pipeline = select_lang_pipeline(splits[3], en_pipeline, ru_pipeline)
            citation_text = preprocess(pipeline, pattern.sub('', pattern_brackets.sub('', splits[3])))
            citation_code = '_'.join(splits[:3])
            context_groups[context_group][citation_code] = citation_text
        except ValueError:
            errors.append(line)
    return context_groups


def pretty_print_topics(topics):
    topics_list = []
    pretty_output = ''
    pretty_topics = [', '.join([re.findall('"([^"]*)"', s)[0] for s in topic[1].split(' + ')]) for topic in topics]
    for i, topic in enumerate(pretty_topics):
        pretty_output += 'Topic {}: {}; '.format(i, topic)
        topics_list.append(topic)
    return pretty_output, topics_list


def print_topics_by_ids(ids, topic_list, ref_key, topics_counts):
    pretty_output = []
    probs = []
    for topic, prob in ids:
        probs.append(round(prob, 2))
        pretty_output.append({'ref_key': ref_key, 'topic': topic_list[topic], 'probability': str(round(prob, 2))})
        topics_counts[topic_list[topic]].append(round(prob, 2))
    return pretty_output, probs, topics_counts


def create_topics(context_groups):
    topics = {}
    topics_dist = defaultdict(lambda: {})
    word_counts = defaultdict(lambda: 0)
    for key, citation in context_groups.items():
        try:
            dictionary = Dictionary(citation.values())
            bow_corpus = [dictionary.doc2bow(doc) for doc in citation.values()]
            lda_model = LdaMulticore(bow_corpus, num_topics=3, id2word=dictionary, passes=2, workers=2)
            topics[key], topics_list = pretty_print_topics(lda_model.print_topics(num_topics=3, num_words=5))
            topics_d = []
            probs = []
            topics_counts = defaultdict(lambda: [])
            for topic in topics_list:
                topic_words = topic.split(', ')
                for word in topic_words:
                    word_counts[word] += 1
            s = 0
            for i in range(len(bow_corpus)):
                pretty_output, probs_, topics_counts = print_topics_by_ids(lda_model[bow_corpus[i]], topics_list,
                                                                           list(citation.keys())[i], topics_counts)
                s += len(pretty_output)
                topics_d.extend(pretty_output)
                probs.extend(probs_)
            topics_counts_ = []
            for key_, value_ in topics_counts.items():
                temp_dict = {}
                temp_dict['topic'] = key_
                temp_dict['number'] = str(len(value_))
                temp_dict['probability_average'] = str(round(np.average(value_), 3))
                temp_dict['probability_std'] = str(round(np.std(value_), 3))
                topics_counts_.append(temp_dict)
            topics_dist[key]['topics'] = sorted(topics_counts_, key=lambda k: int(k['number']), reverse=True)
            topics_dist[key]['contexts'] = sorted(topics_d, key=lambda k: float(k['probability']), reverse=True)
            #         visdata = pyLDAvis.gensim.prepare(lda_model, bow_corpus, dictionary)
        #         pyLDAvis.save_html(visdata, path.join('..', 'data', 'new_vis', '{}_vis.html'.format(key)))
        except ValueError:
            continue
    return dict(topics_dist)


def write_topics(topics_dist, output_dir, file_name='topic_output.json'):
    with open(path.join(output_dir, file_name), 'w') as f:
        dump(dict(topics_dist), f, ensure_ascii=False)


def get_tf_idf_weights(topics):
    vectorizer = TfidfVectorizer(min_df=0, )
    X = vectorizer.fit_transform(topic.replace(', ', ' ') for topic in topics)
    idf = vectorizer._tfidf.idf_
    tf_idf_weights = {}
    for word, weight in dict(zip(vectorizer.get_feature_names(), idf)).items():
        tf_idf_weights[word] = round(weight, 2)
    return tf_idf_weights


def get_counts(topics):
    return Counter(', '.join(topics).split(', '))


def get_topics(topics_):
    topics = []
    for topic_ in topics_['contexts']:
        topics.append(topic_['topic'])
    return topics


def get_words_dict(tf_idf_weights, counts):
    words = defaultdict(lambda: {})
    for word in counts.keys():
        try:
            words[str(word)]['tf_idf'] = float(tf_idf_weights[word])
            words[str(word)]['freq'] = float(counts[word])
        except KeyError:
            pass
    return dict(words)


def get_word_frequencies(topics_dist):
    words_data = defaultdict(lambda: {})
    for key, item in topics_dist.items():
        topics = get_topics(item)
        words_data[key] = get_words_dict(get_tf_idf_weights(topics), get_counts(topics))
    return words_data


def write_word_frequencies(words_data, output_dir, file_name='words_freqs.json'):
    with open(path.join(output_dir, file_name), 'w') as f:
        dump(dict(words_data), f, ensure_ascii=False)


if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('-i', '--input',
                      help="Path to the input file", default="./../../data/citcon4bundles.en.txt")
    parser.add_option('-o', '--output',
                      help="Path to the output directory", default="data")
    options, args = parser.parse_args()
    if not path.exists(options.output):
        makedirs(options.output)
    q = create_context_groups(read_data(options.input))
    print(q['sydDYC'])
    # topics = create_topics(create_context_groups(read_data(options.input)))
    # write_topics(topics, options.output)
    # word_freqs = get_word_frequencies(topics)
    # write_word_frequencies(word_freqs, options.output)
