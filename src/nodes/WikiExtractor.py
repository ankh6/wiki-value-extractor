from typing import List
from interfaces.Extractor import Extractor
from utils.ExtractorUtils import set_custom_preprocessor
from clients.Data import customer_queries, url
from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes import BM25Retriever, Crawler, FARMReader, TextConverter
from haystack.pipelines import ExtractiveQAPipeline, Pipeline
from haystack.schema import Answer, Document, EvaluationResult, Label, MultiLabel
from haystack.utils import launch_es, print_answers

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
        ''' Reads a web page and writes the page content to a Store

        Arguments:
        source_files, a list of web pages to be crawled

        Returns:

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
            cleaned_docs = processor.process(document_to_process)
            # Writes documents to the Elastic store
            self.write_document_to_store(cleaned_docs)
            # Predicts answers given queries
            predicted_answers = self.predict_answers_given_queries(customer_queries)
            # Converts file content into a Question Answering format
            self.convert_content_to_QA_format(customer_queries,predicted_answers)
        except Exception as e :
            print(f"Something wrong happened ! {e.args}")
            raise e
        finally:
            print("Crawled content at ./crawled_files path")
            
    def write_document_to_store(self, documents):
        ''' Indexes documents to the ElasticSearch for later queries

        Arguments:
        document, a list of documents to index into the ElasticSearch document store
        '''
        store = self.get_store()
        store.write_documents(documents)

    def predict_answers_given_queries(self, queries):
        ''' Predicts answers given queries of the customer

        Arguments:
        queries, a list of queries that the customer is interested in

        Returns:
        predicted_answers, a list of predicted answers. Same order in which the queries were given
        
        '''
        store = self.get_store()
        retriever = BM25Retriever(document_store=store)
        reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=False)
        pipe = ExtractiveQAPipeline(reader,retriever)
        predicted_answers = []
        for single_query in queries:
            prediction = pipe.run(query=single_query,params={"Retriever":{"top_k": 1}, "Reader":{"top_k": 1}}, debug=True)
            predicted_answers.append(prediction["answers"])
        return predicted_answers

    def convert_content_to_QA_format(self, customer_queries, predicted_answers):
        print("Received predicted answers: ", predicted_answers)
        # Customer queries and answers predicted by the model have the same index, i.e. order
        # Extracts the index of the query in its data structure
        for query, answer in enumerate(predicted_answers):
            answer = answer[0]
            text_to_QA_format = {
                "data" : [
                    {
                        "title" : "Relevant information",
                        "paragraphs": [
                            {
                                "id" : answer.document_id,
                                "query" : customer_queries[query],
                                "answers" : [{"text": answer.answer, "answer_start": answer.offsets_in_document[0].start, "answer_end" : answer.offsets_in_document[0].end}],
                                "context" : answer.context,
                                "is_correct_answer" : answer.score
                            }
                        ]
                    }
                ]
            }