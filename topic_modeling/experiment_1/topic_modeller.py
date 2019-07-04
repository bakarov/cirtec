from gensim.models import LdaMulticore
from gensim.corpora import Dictionary
from parser import Parser
from preprocesser import Preprocesser
from os import path, makedirs, listdir
from ufal.udpipe import Model, Pipeline
import re
import optparse


def pretty_print_topics(topics):
    topics_list = []
    pretty_output = ''
    pretty_topics = [', '.join([re.findall('"([^"]*)"', s)[0] for s in topic[1].split(' + ')]) for topic in topics]
    for i, topic in enumerate(pretty_topics):
        pretty_output += 'Topic {}: {}; '.format(i, topic)
        topics_list.append(topic)
    return pretty_output, topics_list


def print_topics_by_ids(ids, topic_list):
    pretty_output = ''
    for topic, prob in ids:
        pretty_output += '{}, probability: {:0.2f}; '.format(topic_list[topic], prob)
    return pretty_output


def extract_topics_from_text(text_tokenized):
    try:
        dictionary = Dictionary(text_tokenized)
        bow_corpus = [dictionary.doc2bow(doc) for doc in text_tokenized]
        lda_model = LdaMulticore(bow_corpus, num_topics=1, random_state=42, id2word=dictionary, workers=2)
        topics, topics_list = pretty_print_topics(lda_model.print_topics(num_topics=1, num_words=5))
    except ValueError:
        return 'No topics extracted'
    return topics


def extract_topics_from_sections(sections, preprocesser, dirname, pipeline, filename='sections_topics.txt'):
    with open(path.join(dirname, filename), 'w') as f:
        for section_key, section_value in sections.items():
            topics = extract_topics_from_text(preprocesser.preprocess(section_value, pipeline))
            f.write('Section: {}\n{}\n\n'.format(section_key, topics))


def extract_topics_from_pars(pars, preprocesser, dirname, pipeline, filename='pars_topics.txt'):
    with open(path.join(dirname, filename), 'w') as f:
        for par_id, par in enumerate(pars):
            topics = extract_topics_from_text(preprocesser.preprocess(par, pipeline))
            f.write('Paragraph: {}\n{}\n\n'.format(par_id, topics))


def extract_topics_from_full_text(full_text, preprocesser, dirname, pipeline, filename='full_text_topics.txt'):
    topics = extract_topics_from_text(preprocesser.preprocess(full_text, pipeline))
    with open(path.join(dirname, filename), 'w') as f:
        f.write(topics)


if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('-i', '--input',
                      help="Path to the input file", default="./../data/papers")
    parser.add_option('-o', '--output',
                      help="Path to the output directory", default="output")
    options, args = parser.parse_args()
    for current_file in listdir(options.input):
        dirname = path.join(options.output, path.splitext(path.split(current_file)[~0])[0])
        if not path.exists(dirname):
            makedirs(dirname)
        parser = Parser(path.join(options.input, current_file))
        parser.load_data()
        sections, pars, full_text = parser.get_parsed_data()
        preprocesser = Preprocesser()
        udpipe_model_path = '../udpipe/english-ewt-ud-2.3.udpipe'
        model = Model.load(udpipe_model_path)
        pipeline = Pipeline(model, 'tokenize', Pipeline.DEFAULT, Pipeline.DEFAULT, 'conllu')
        # extract_topics_from_sections(sections, preprocesser, dirname, pipeline)
        extract_topics_from_pars(pars, preprocesser, dirname, pipeline)
        extract_topics_from_full_text(full_text, preprocesser, dirname, pipeline)
