from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import heapq
from string import punctuation
from transformers import BartForConditionalGeneration, BartTokenizer

def hybrid_text_summarizer(text, extractive_ratio=0.5):
    """
    Generates a summary of the text using a hybrid of extractive and abstractive techniques.
    
    :param text: The text to summarize
    :param extractive_ratio: The ratio of extractive sentences to include in the summary
    :return: A string that represents the summary of the text
    """
    # Extractive Summarization
    sentences = sent_tokenize(text)
    stop_words = set(stopwords.words("english") + list(punctuation))
    words = word_tokenize(text.lower())
    filtered_words = [word for word in words if word not in stop_words]
    word_frequencies = FreqDist(filtered_words)
    num_extractive_sentences = max(1, int(len(sentences) * extractive_ratio))
    sentence_scores = {}
    for sentence in sentences:
        for word in word_tokenize(sentence.lower()):
            if word in word_frequencies:
                if sentence not in sentence_scores:
                    sentence_scores[sentence] = word_frequencies[word]
                else:
                    sentence_scores[sentence] += word_frequencies[word]
    summary_sentences = heapq.nlargest(num_extractive_sentences, sentence_scores, key=sentence_scores.get)
    extractive_summary = ' '.join(summary_sentences)

    # Abstractive Summarization
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
    inputs = tokenizer([text], max_length=1024, return_tensors='pt', truncation=True)
    summary_ids = model.generate(inputs['input_ids'], num_beams=4, min_length=30, max_length=100, early_stopping=True)
    abstractive_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # Combine Extractive and Abstractive Summaries
    combined_summary = f"{abstractive_summary}\n\nExtractive Summary:\n{extractive_summary}"
    
    return combined_summary

# Example usage
text = """
"""
summary = hybrid_text_summarizer(text)
print(summary)