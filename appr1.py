import dask.dataframe as dd
import dask.array as da
import dask_ml.feature_extraction.text as dask_text
from dask_ml.decomposition import TruncatedSVD
import dask
from dask import delayed
from dask import compute
from dask.distributed import Client
from dask_ml.feature_extraction.text import HashingVectorizer
import dask_ml
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np

class pdf_vector():
    
    def __init__(self,df):
        self.df = df
        self.final()


    #TF-IDF
    def tfidf_vector(self,text):
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([text])

        return tfidf_matrix.toarray()[0]


    
    def rnn(self,text):

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts([text])
        sequences = tokenizer.texts_to_sequences([text])

        # Pad sequences
        max_length = max([len(s) for s in sequences])
        padded_sequences = pad_sequences(sequences, maxlen=max_length)

   
        model = Sequential()
        model.add(Embedding(len(tokenizer.word_index) + 1, 100, input_length=max_length))
        model.add(LSTM(100))
        model.add(Dense(1))


        model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

  
        target_data = np.random.rand(padded_sequences.shape[0],1)

        model.fit(padded_sequences, target_data, epochs=10, verbose=0,batch_size=32, callbacks=[EarlyStopping(monitor='loss', patience=3)])


        embedding = model.get_weights()[0]

        return embedding.flatten()
       

    def bert(self,text):

        # Load the BERT model and tokenizer
        model = BertModel.from_pretrained('bert-base-uncased')
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        #maximum length of input text
        max_length = 512

        # Tokenize the text
        tokenized_text = tokenizer.tokenize(text)

        # Truncate the text if it is longer than the maximum length
        if len(tokenized_text) > max_length:
            tokenized_text = tokenized_text[:max_length]

        # Convert the tokenized text to BERT inputs
        bert_inputs = tokenizer.convert_tokens_to_ids(tokenized_text)

        # Pass the inputs through the BERT model to get the embeddings
        embeddings = model(torch.tensor(bert_inputs).unsqueeze(0))[0]

        # Extract the embedding vector
        pdf_vector = embeddings.mean(dim=0)
        pdf_vector = pdf_vector.detach().numpy()
        
        return pdf_vector.flatten()
    
 
        # Convolutional Neural Networks (CNNs):
    def cnn(self,sentences):
        # One-hot encode the sentences
        vocab_size = 50
        encoded_docs = [one_hot(d, vocab_size) for d in sentences]

        # Pad the encoded documents to the same length
        max_length = 4
        padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding="post")

        # Define the model
        model = Sequential()
        model.add(Embedding(vocab_size, 8, input_length=max_length))
        model.add(Conv1D(32, 3, activation="relu"))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


        target_data = np.random.randint(0, 2, (len(sentences), 1))


        model.fit(padded_docs, target_data, epochs=10, verbose=0, batch_size=32, callbacks=[EarlyStopping(monitor='accuracy', patience=3)])


        embedding = model.layers[0].get_weights()[0]

        return embedding.flatten()

        
    # Latent Dirichlet Allocation (LDA)
    def lda(self,sentences):
  
        texts = [doc.split() for doc in sentences]
        dictionary = Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]
        model = gensim.models.LdaModel(corpus, num_topics=3, id2word=dictionary)
        vectors = [model[doc] for doc in corpus]

        return vectors.flatten()

         # Non-Negative Matrix Factorization (NMF)
    def nmf(self,corpus):


        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(corpus)

        nmf = NMF(n_components=3, init='random', random_state=0)
        W = nmf.fit_transform(X)

        return W.flatten()

    # Latent Semantic Analysis (LSA)
    def lsa(self,docs):

        vectorizer = CountVectorizer(stop_words='english')
        doc_word = vectorizer.fit_transform(docs)

        lsa = TruncatedSVD(n_components=2)
        doc_topic = lsa.fit_transform(doc_word)

        # Show the results
  
        #print(doc_topic)#Document-Topic Matrix:
   
        #print(lsa.components_) #LSA Components#used for clustering
        vector=lsa.components_
        return vector.flatten()



    #doc2vec
    def doc2vec(self,documents):

        tagged_documents = [gensim.models.doc2vec.TaggedDocument(doc.split(), [i]) for i, doc in enumerate(documents)]

        model = gensim.models.Doc2Vec(tagged_documents, vector_size=100, window=5, min_count=1, workers=4)

        doc_vectors = model.docvecs

        return doc_vectors[0]

    
    def get_sentences(self,text):
            sentences = nltk.sent_tokenize(text)
            return sentences


    def final(self):

        self.df['TF-IDF vectors']=self.df['pdftext'].apply(self.tfidf_vector)
        self.df['RNN vectors']=self.df['pdftext'].apply(self.rnn)
        self.df['BERT vectors']=self.df['pdftext'].apply(self.bert)
        self.df['sentences']=self.df['pdftext'].apply(self.get_sentences)
        self.df['CNN vectors']=self.df['sentences'].apply(self.cnn)
        self.df['lda vectors']=self.df['sentences'].apply(self.lda)
        self.df['nmf vectors']=self.df['sentences'].apply(self.nmf)
        self.df['lsa vectors']=self.df['sentences'].apply(self.lsa)
        self.df['doc2vec vectors']=self.df['sentences'].apply(self.doc2vec)

      

        return self.df
