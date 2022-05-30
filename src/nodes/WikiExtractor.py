from typing import List
from interfaces.Extractor import Extractor
from utils.ExtractorUtils import set_custom_preprocessor
from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes import BM25Retriever, Crawler, FARMReader, TextConverter
from haystack.pipelines import ExtractiveQAPipeline
from haystack.schema import Answer, Document, EvaluationResult, Label, MultiLabel
from haystack.utils import launch_es, print_answers

url = ["https://en.wikipedia.org/w/index.php?title=Databricks&oldid=1077382609"]

class WikiExtractor(Extractor):
    def __init__(self):
        self.converter = TextConverter(valid_languages="en")
        self.crawler = Crawler(output_dir="crawled_files", crawler_depth=0, overwrite_existing_files=True)
        self.store = ElasticsearchDocumentStore(host="localhost", index="document", duplicate_documents="overwrite")

    def get_converter(self):
        return self.converter
    
    def get_crawler(self):
        return self.crawler
    
    def get_store(self):
        return self.store
    
    def read_page(self, source_files: List[str]):
        crawler = self.get_crawler()
        converter = self.get_converter()
        try:
            # Returns a list of paths that will be processed downstream
            file_paths = crawler.crawl(urls=source_files)
            # Convert the file path i.e. source at the first index to a Document type
            document_to_process = converter.convert(file_paths[0])
            processor = set_custom_preprocessor(clean_whitespace=True, clean_empty_lines=True, split_by="word", language="en")
            # Returns a list of processed documents
            cleaned_docs = processor.process(document_to_process)
            #print(f"Doc {cleaned_docs[0].content}, {cleaned_docs[0].meta}")
            self.write_document_to_store(cleaned_docs)
            predicted_answers = self.test_pipeline(cleaned_docs)
            self.convert_content_to_QA_format(predicted_answers)
        except Exception as e :
            print(e, (e.args))
            raise e
        finally:
            print("Crawled content at ./crawled_files")
            
    def write_document_to_store(self, document):
        store = self.get_store()
        store.write_documents(document)

    def test_pipeline(self, documents=None):
        store = self.get_store()
        retriever = BM25Retriever(document_store=store)
        reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=False)
        pipe = ExtractiveQAPipeline(reader,retriever)
        #result = reader.predict(query="What is the revenue?", documents=documents, top_k = 5)
        prediction = pipe.run(query="What is the revenue?", params={"Retriever":{"top_k": 1}, "Reader":{"top_k": 5}}, debug=True)
        print("Prediction to query: ", prediction,  "Type: ", prediction["answers"])
        return prediction["answers"]
        #pipe.print_eval_report(eval_result=prediction, document_scope="answer",answer_scope="document_id_and_context")

    def convert_content_to_QA_format(self, answers):
        most_relevant_information = answers[0]
        text_to_QA_format = {
            "data" : [
                {
                    "title" : "Clients labels",
                    "paragraphs": [
                        {
                            "context" : most_relevant_information.context,
                            "id" : most_relevant_information.document_id,
                            "answers" : [{"text": most_relevant_information.answer, "answer_start": most_relevant_information.offsets_in_document[0].start, "answer_end" : most_relevant_information.offsets_in_document[0].end}],
                            "is_correct_answer" : most_relevant_information.score,
                        }
                    ]
                }
            ]
        }
        return text_to_QA_format
