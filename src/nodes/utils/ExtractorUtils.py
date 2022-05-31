from haystack.nodes import PreProcessor

predictive_model = "deepset/roberta-base-squad2"

def set_custom_preprocessor(clean_whitespace=True, clean_empty_lines=True, split_by="word", language="en"):
    return PreProcessor(clean_empty_lines=clean_empty_lines, clean_whitespace=clean_whitespace, clean_header_footer=False, split_by=split_by, language=language, split_respect_sentence_boundary=True)