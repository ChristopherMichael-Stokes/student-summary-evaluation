import logging
import os
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Set, Union

import datasets
import nltk
import numpy as np
import pandas as pd
from nltk import ngrams
from nltk.corpus import stopwords, words
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')
nltk.download('words')

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def text_tokenize(text: str, stop_words: Set[str], lemmatiser: WordNetLemmatizer) -> List[str]:
    """Tokenise the input text

    Args:
        text (str): The input document to tokenise
        stop_words (Set[str]): The set of stopwords to exclude
        lemmatiser (WordNetLemmatizer): Object to use for lemmatisation

    Returns:
        List[str]: List of tokens
    """
    return [lemmatiser.lemmatize(tok.lower()) for tok in word_tokenize(text) if tok.isalnum() and tok not in stop_words]


def make_bigram(tokens: Union[List[str], np.ndarray]) -> Set[str]:
    """Create all possible unique bigrams from input tokens

    Args:
        tokens (Union[List[str], np.ndarray]): Tokens from a document in the order they originally appeared pre-tokenisation

    Returns:
        Set[str]: Unique bigrams
    """
    if type(tokens) == np.ndarray:
        tokens = tokens.tolist()
    return set(ngrams(tokens, 2))


def nlp_preprocess(df: pd.DataFrame, stop_words: Set[str], lemmatiser: WordNetLemmatizer, column: str):
    """Preprocess single column from a supplied dataframe and adds new features inplace

    Args:
        df (pd.DataFrame): DataFrame containing column to process
        stop_words (Set[str]): Stopwords to exclude
        lemmatiser (WordNetLemmatizer): Object to use for lemmatisation
        column (str): Name of the column being processed
    """
    f_tok = partial(text_tokenize, stop_words=stop_words, lemmatiser=lemmatiser)
    df[f'{column}_lemmas'] = df[column].apply(f_tok)
    df[f'{column}_bigram'] = df[f'{column}_lemmas'].apply(make_bigram)


def process_col(data: List[str], stop_words: Set[str], lemmatiser: WordNetLemmatizer, col: str) -> Dict[str, Any]:
    """Preprocess single column given as a list of strings

    Args:
        data (List[str]): Column to process
        stop_words (Set[str]): Stopwords to exclude
        lemmatiser (WordNetLemmatizer): Object to use for lemmatisation
        col (str): Name of the column being processed (only used for naming)

    Returns:
        Dict[str, Any]: New features resulting from the preprocessing
    """
    lemmas = [text_tokenize(row, stop_words, lemmatiser) for row in data]
    bigrams = [make_bigram(lemma) for lemma in lemmas]
    return {f'{col}_lemmas': lemmas, f'{col}_bigram': bigrams}


def add_bigram_features(df: pd.DataFrame):
    norm_scale = df.text_unique_bigrams

    df['text_bigram_overlap'] = df[['prompt_text_bigram', 'text_bigram']].apply(
        lambda row: len(row[0] & row[1]), axis=1) / norm_scale
    df['question_bigram_overlap'] = df[['prompt_question_bigram', 'text_bigram']].apply(
        lambda row: len(row[0] & row[1]), axis=1) / norm_scale
    df['text_bigram_ratio'] = df['text_unique_bigrams'] / (df['prompt_text_unique_bigrams'])

    df['text_bigram_diff'] = df[['prompt_text_bigram', 'text_bigram']].apply(
        lambda row: len(row[1] - row[0]), axis=1) / norm_scale
    df['question_bigram_diff'] = df[['prompt_question_bigram', 'text_bigram']].apply(
        lambda row: len(row[1] - row[0]), axis=1) / norm_scale

    df['text_bigram_exclusive'] = df[['prompt_text_bigram', 'text_bigram']].apply(
        lambda row: len(row[0] ^ row[1]), axis=1) / norm_scale
    df['question_bigram_exclusive'] = df[['prompt_question_bigram', 'text_bigram']].apply(
        lambda row: len(row[0] ^ row[1]), axis=1) / norm_scale


def add_word_features(df: pd.DataFrame):
    df['n_words'] = df.text_lemmas.str.len()
    df['unique_words'] = df.text_lemmas.apply(set).str.len()
    df['unique_ratio'] = df.unique_words / df.n_words

    df['word_lengths'] = df.text_lemmas.apply(lambda x: [len(y) for y in x])
    df['word_len_avg'] = df.word_lengths.apply(np.mean)

    df['word_len_q10'] = df.word_lengths.apply(partial(np.percentile, q=10))
    df['word_len_q90'] = df.word_lengths.apply(partial(np.percentile, q=90))


def pos_counts(tags):
    dd = defaultdict(lambda: 0)
    for _, pos in tags:
        dd[pos] += 1
    return dd


def add_pos_features(df: pd.DataFrame):
    df['pos'] = df.text_lemmas.apply(partial(pos_tag, tagset='universal'))
    df['pos_counts'] = df.pos.apply(pos_counts)

    df['verb_count'] = df.pos_counts.str['VERB'].replace(np.nan, 0)
    df['noun_count'] = df.pos_counts.str['NOUN'].replace(np.nan, 0)
    df['adv_count'] = df.pos_counts.str['ADV'].replace(np.nan, 0)
    df['adj_count'] = df.pos_counts.str['ADJ'].replace(np.nan, 0)
    df['det_count'] = df.pos_counts.str['DET'].replace(np.nan, 0)


def make_split(summaries_path: Path, prompts_path: Path, make_pos_features: bool) -> pd.DataFrame:
    """Load and process a dataset split from the given input files

    Args:
        summaries_path (Path): Path to summaries csv
        prompts_path (Path): path to prompts csv

    Returns:
        pd.DataFrame: Built dataset
    """
    stop_words = set(stopwords.words('english'))
    lemmatiser = WordNetLemmatizer()

    # Load base csv
    log.info('Loading data files')
    summaries_df = pd.read_csv(summaries_path)
    prompts_df = pd.read_csv(prompts_path)

    # Preprocess prompts in pandas as data is very small
    log.info('Preprocess prompt data')
    for column in ['prompt_title', 'prompt_question', 'prompt_text']:
        nlp_preprocess(prompts_df, stop_words, lemmatiser, column)
        prompts_df[f'{column}_unique_bigrams'] = prompts_df[f'{column}_bigram'].str.len()

    # Use huggingface arrow dataset to process summaries in parallel batches
    log.info('Preprocess summaries data')
    summaries_dataset = datasets.Dataset.from_pandas(summaries_df, preserve_index=False)
    proc_func = partial(process_col, stop_words=stop_words, lemmatiser=lemmatiser, col='text')
    summaries_df = summaries_dataset.map(function=lambda example: {
                                         **proc_func(example['text']), **example}, num_proc=os.cpu_count(), keep_in_memory=True, batched=True).to_pandas()
    summaries_df['text_bigram'] = summaries_df.text_bigram.apply(lambda row: {(x[0], x[1]) for x in row})
    summaries_df['text_unique_bigrams'] = summaries_df['text_bigram'].str.len()

    # Using left join in the rare occurence we have a summary with incorrect prompt id
    log.info('Join prompts to summaries')
    df = pd.merge(summaries_df, prompts_df, how='left', on='prompt_id')
    df.fillna('')

    # Create new features
    log.info('Adding bigram features')
    add_bigram_features(df)
    log.info('Adding word based features')
    add_word_features(df)
    if make_pos_features:
        log.info('Adding POS based features')
        add_pos_features(df)

    return df


if __name__ == '__main__':
    data_dir = Path('data/commonlit-evaluate-student-summaries')

    sample_submission = data_dir / 'sample_submission.csv'
    summaries_train = data_dir / 'summaries_train.csv'
    summaries_test = data_dir / 'summaries_test.csv'
    prompts_train = data_dir / 'prompts_train.csv'
    prompts_test = data_dir / 'prompts_test.csv'

    test = False
    if test:
        df = make_split(summaries_path=summaries_test, prompts_path=prompts_test, make_pos_features=True)
    else:
        df = make_split(summaries_path=summaries_train, prompts_path=prompts_train, make_pos_features=True)

    pass
