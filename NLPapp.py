import streamlit as st
import spacy
#python -m spacy download en
from textblob import TextBlob
from gensim.summarization import summarize

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer





def sumy_summarizer(docx):
    parser = PlaintextParser.from_string(docx, Tokenizer("english"))
    lex_summarizer = LexRankSummarizer()
    summary = lex_summarizer(parser.document,3)
    summary_list = [str(sentence) for sentence in summary]
    result = ' '.join(summary_list)
    return result

def text_analyzer(my_text):
    nlp = spacy.load('en')
    docx = nlp(my_text)

    tokens = [token.text for token in docx]
    allData = [('"Tokkens":{},\n"Lemma":{}'.format(token.text,token.lemma_)) for token in docx]
    return allData

def entity_analyzer(my_text):
    nlp = spacy.load('en')
    docx = nlp(my_text)
    entities = [(entity.text,entity.label_) for entity in docx.ents ]
    return entities

def main():
    
    st.sidebar.title("Natural language processing")
    st.sidebar.subheader('Which number do you like best?')
    
    st.title("NLP System Using Streamlit")
    st.subheader("Natural Language Processing on Go")

    sliderSlectBox = st.sidebar.selectbox('How would you like to be contacted?',
        ('Tokenization', 'Named Entity', 'Sentiment Analysis','Text Summerization'))

    #Tokenization
    if sliderSlectBox == "Tokenization":
        st.subheader("Toakkenize your text.")
        message = st.text_area("Enter your text here : ")
        if st.button("Analysis"):
            nlp_result = text_analyzer(message)
            st.json(nlp_result)
    #Named Entity
    if sliderSlectBox == "Named Entity":
        st.subheader("Extract entities from your text.")
        message = st.text_area("Enter your text here : ")
        if st.button("Extract"):
            nlp_result = entity_analyzer(message)
            st.json(nlp_result)

    #Sentiment Analysis
    if sliderSlectBox == "Sentiment Analysis":
        st.subheader("Sentiment of your text.")
        message = st.text_area("Enter your text here : ")
        if st.button("Analysis"):
            blob = TextBlob(message)
            result_sentiment = blob.sentiment
            st.success(result_sentiment)
    #Text Summerization
    if sliderSlectBox == "Text Summerization":
        st.subheader("Summerize your text.")
        message = st.text_area("Enter your text here : ")
        summary_options = st.selectbox("Choice your Summerize",("Gensim","Sumy"))
        if st.button("Analysis"):
            if summary_options == "Gensim":
                st.text("Using Gensim...")
                summary_result = summarize(message)
            elif summary_options == "Sumy":
                st.text("Using Sumy...")
                summary_result = sumy_summarizer(message)
            else:
                st.warning("Using default Summarizer")
                st.text("Using Gensim")
                summary_result = summarize(message)
            
            st.success(summary_result)


            
if __name__ == "__main__":
    main()