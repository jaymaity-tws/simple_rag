import json
import os
from getpass import getpass
from urllib.request import urlopen

import nest_asyncio
import openai
import pandas as pd
import phoenix as px
from gcsfs import GCSFileSystem
from llama_index.core import (
    Settings,
    StorageContext,
    load_index_from_storage,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from phoenix.evals import (
    HallucinationEvaluator,
    OpenAIModel,
    QAEvaluator,
    RelevanceEvaluator,
    run_evals,
)
from phoenix.session.evaluation import get_qa_with_reference, get_retrieved_documents
from phoenix.trace import DocumentEvaluations, SpanEvaluations
from tqdm import tqdm

nest_asyncio.apply()  # needed for concurrent evals in notebook environments
pd.set_option("display.max_colwidth", 1000)

queries_df = get_qa_with_reference(px.active_session())
retrieved_documents_df = get_retrieved_documents(px.active_session())

eval_model = OpenAIModel(
    model="gpt-4o",
)
hallucination_evaluator = HallucinationEvaluator(eval_model)
qa_correctness_evaluator = QAEvaluator(eval_model)
relevance_evaluator = RelevanceEvaluator(eval_model)

hallucination_eval_df, qa_correctness_eval_df = run_evals(
    dataframe=queries_df,
    evaluators=[hallucination_evaluator, qa_correctness_evaluator],
    provide_explanation=True,
)
relevance_eval_df = run_evals(
    dataframe=retrieved_documents_df,
    evaluators=[relevance_evaluator],
    provide_explanation=True,
)[0]

px.Client(endpoint="http://127.0.0.1:6008/").log_evaluations(
    SpanEvaluations(eval_name="Hallucination", dataframe=hallucination_eval_df),
    SpanEvaluations(eval_name="QA Correctness", dataframe=qa_correctness_eval_df),
    DocumentEvaluations(eval_name="Relevance", dataframe=relevance_eval_df),
)