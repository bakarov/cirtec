from collections import defaultdict
from os import path


class Parser:
    def __init__(self, input_path):
        self.lines = []
        self.pars = ''
        self.sections = defaultdict(lambda: '')
        self.full_text = ''
        self.input_path = input_path

    def load_data(self):
        with open(self.input_path, 'r') as f:
            self.lines = f.read().split('\n')[:-1]
        self.extract_sections()
        self.extract_pars()
        self.extract_full_text()

    def extract_sections(self, section_marker='C', split_marker=':', marker_id=-2, text_id=-1):
        current_section_key = ''
        for line in self.lines:
            metadata = line.split(split_marker)
            if metadata[marker_id] == section_marker:
                current_section_key = metadata[text_id].strip()
            else:
                self.sections[current_section_key] += metadata[text_id]

    def extract_pars(self, par_splitter=':A:', par_marker='A', split_marker=':', marker_id=-2, text_id=-1):
        for line in self.lines:
            metadata = line.split(split_marker)
            if not metadata[text_id].strip():
                continue
            if metadata[marker_id] == par_marker:
                self.pars += par_splitter
            else:
                self.pars += metadata[text_id]
        self.pars = self.pars.split(par_splitter)

    def extract_full_text(self, section_marker='C', split_marker=':', marker_id=-2, text_id=-1):
        for line in self.lines:
            metadata = line.split(split_marker)
            if metadata[marker_id] != section_marker:
                self.full_text += metadata[text_id]

    def get_parsed_data(self):
        return self.sections, self.pars, self.full_text
