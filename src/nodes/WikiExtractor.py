from typing import List
from interfaces.Extractor import Extractor
from utils.ExtractorUtils import set_custom_preprocessor, predictive_model
from clients.Data import customer_queries, url
from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes import BM25Retriever, Crawler, FARMReader, TextConverter
from haystack.pipelines import ExtractiveQAPipeline
from haystack.schema import Answer, Document
from haystack.utils import launch_es


class WikiExtractor(Extractor):
    def __init__(self):
        self.converter = TextConverter(valid_languages="en")
        self.crawler = Crawler(output_dir="crawled_files", crawler_depth=0, overwrite_existing_files=True)
        # Spins up an Elastic instance.
        # Throws a ConnectionError if connection fails
        launch_es()
        self.store = ElasticsearchDocumentStore(host="localhost", port="9200", index="document", duplicate_documents="overwrite")

    def get_converter(self):
        return self.converter
    
    def get_crawler(self):
        return self.crawler
    
    def get_store(self):
        return self.store
    
    def extract_content(self, source_files: List[str], customer_queries: List[str], predictive_model, max_answers : int =1):
        ''' Main Routine
        Extracts the content of a Wikipedia page and provides an answer to the queries, the context,
        the start and end offsets of the answer and a confidence score ranging from 0 to 1

        By default, the model uses the roberta-base-squad2 for its Reader component.

        Arguments:
        source_files, a list of Wikipedia pages
        customer_queries, a list of queries to be answered in the document. Given by the customer
        max_answers, an integer. It defines the number of neighbours that the model will rely on for prediction
        '''
        crawler = self.get_crawler()
        converter = self.get_converter()
        try:
            # Returns a list of paths that will be processed downstream
            file_paths = crawler.crawl(urls=source_files)
            # Generates a Haystack list of Documents from file paths
            document_to_process = converter.convert(file_paths[0])
            # Processes the list of Documents
            processor = set_custom_preprocessor(clean_whitespace=True, clean_empty_lines=True, split_by="word", language="en")
            # Returns a list of processed documents given the process rules
            cleaned_documents = processor.process(document_to_process)
            # Writes documents to the Elastic store
            self.write_document_to_store(cleaned_documents)
            # Predicts answers given queries
            inferred_answers = self.infer_answers_given_queries(customer_queries, predictive_model, max_answers)
            # Converts file content into a Question Answering format
            self.convert_content_to_QA_format(customer_queries,inferred_answers)
        except Exception as e :
            print("Something wrong happened !")
            raise e
        finally:
            print("Content file can be found at ./crawled_files folder")
            
    def write_document_to_store(self, documents: List[Document]):
        ''' Indexes documents to the ElasticSearch for later queries

        Arguments:
        document, a list of documents to index into the ElasticSearch document store
        '''
        store = self.get_store()
        store.write_documents(documents)

    def infer_answers_given_queries(self, customer_queries: List[str], predictive_model, max_answers : int):
        ''' Predicts answers given customer queries

        Arguments:
        customer_queries, a list of queries to be answered in the document. Given by the customer
        max_answers, the maximum number of answers that the model provides

        Returns:
        inferred_answers, a list of inferred answers. The answers are in the same order in which the queries were given
        '''
        store = self.get_store()
        retriever = BM25Retriever(document_store=store)
        reader = FARMReader(model_name_or_path=predictive_model, use_gpu=False)
        pipe = ExtractiveQAPipeline(reader,retriever)
        inferred_answers = []
        for single_query in customer_queries:
            prediction = pipe.run(query=single_query,params={"Retriever":{"top_k": 1}, "Reader":{"top_k": max_answers}}, debug=True)
            inferred_answers.append(prediction["answers"])
        return inferred_answers

    def convert_content_to_QA_format(self, customer_queries: List[str], inferred_answers: List[Answer]):
        ''' Converts the inferred answers into a format that is suitable for Haystack downstream tasks

        Arguments:
        customer_queries, a list of queries to be answered in the document. Given by the customer
        inferred_answers, a list of inferred answers. The answers are in the same order in which the queries were given
        
        '''
        # Customer queries and answers inferred by the model have the same index, i.e. order
        # Extracts the index of the query in its data structure
        for query_index, answer in enumerate(inferred_answers):
            answer = answer[0]
            text_to_QA_format = {
                "data" : [
                    {
                        "title" : "Relevant information",
                        "paragraphs": [
                            {
                                "qas": [
                                    {
                                "document_id" : answer.document_id,
                                "query" : customer_queries[query_index],
                                "answers" : [{"text": answer.answer, "context":answer.context, "answer_start": answer.offsets_in_document[0].start, "answer_end" : answer.offsets_in_document[0].end}],
                                "confidence_score" : answer.score }
                                ]
                            }
                        ]
                    }
                ]
            }
            print(text_to_QA_format)


if __name__ == '__main__':
    wiki_extractor = WikiExtractor()
    wiki_extractor.extract_content(url, customer_queries, predictive_model, max_answers=1)