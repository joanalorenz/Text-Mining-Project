# Import utils
from utils import *

#model = KeyedVectors.load_word2vec_format(r'GoogleNews-vectors-negative300.bin', binary=True) # load Google Model
model = gensim.models.Word2Vec.load('mymodel') # Load own model, trained on 1 Mio News Dataset

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# ----------------  Set up database for model comparison --------------------
  
# Import All the News Kaggle Dataset (used for ploting and model comparison)
#database = r'articles.db'
#df = pd.concat([pd.read_csv(file) for file in glob.glob("articles*.csv")], ignore_index = True)
#df = df[['title','content']]
#df = df[(df['content'].str.len()>=2800) & (df['title'].str.len()>=70)]
#  
#
##  Sample the dataframe
#sample_size=200
#if df.shape[0] > sample_size:
#     df = df[0:sample_size]
##Create the SQL database
#sql = SQL(database)
#sql.create_database('articles')
#sql.import_data('articles', df)

    
# Sample the database
#database = r'articles.db'
#sql = SQL(database)
#x = sql.get_sample(150)

# Webscrape a URL
url='https://www.forbes.com/sites/chloedemrovsky/2019/04/15/notre-dame-in-flames-protecting-our-cultural-treasures/#581c617c5f10'
x = pd.DataFrame(columns=['content', 'title'])
x.loc[0,['content', 'title']] = get_article(url)

# Remove multiple whitespaces
for index, row in x.iterrows():
    row['content']=' '.join(row['content'].split())

# Create corpus dictionary
dictionary = corpora.Dictionary()

# Define number of top sentences to pick for each topic
num_of_sents = 3
num_of_char = 250
# Define quantile of similarity score the sentences need to fulfill
quantile= 0.85

# run functions for each article in the corpus
for index, row in x.iterrows():
    # Progress tracker
    print("Summarizing article ",index +1, "out of", x.shape[0])
    article = row['content']
    # process the content
    sentences = split_into_sentences(row['content'])
    # preprocess the sentences (cleaning, tokenization, remove stopwords)
    article_clean = [process_raw(sentence) for sentence in sentences]
    # assign cleaned content to dataframe
    x.loc[index,'content_clean'] = row['content']
    x.loc[index,'content_clean'] = article_clean
    
    # creating bag of words for gensim library
    dictionary.add_documents(article_clean)
        
    # create corpus in desired format  
    corpus = [dictionary.doc2bow(sentence) for sentence in article_clean]
    
    # create instance of a class to perform tfidf
    tfidf = models.TfidfModel(corpus)
    
    # perform tfidf
    corpus_tfidf = tfidf[corpus]
    
    # print sample of corpus after performing tfidf
    count = 0

    #--------------- perform LSI -----------------
    lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=3)
    corpus_lsi = lsi[corpus_tfidf]
    
    V = matutils.corpus2dense(lsi[corpus_tfidf], len(lsi.projection.s)).T / lsi.projection.s
    
    b = np.array(V)
    # create a list of the similarities of each sentence with the topics
    b_list = [list(b[:, i - 1]) for i in range(b.shape[1])]
    
    
    # save the length and the similarity scores, ordered by the latter 
    sims_ = []
    lengths = []
    for i in b_list:
        tuples = [(i.index(value), abs(value), len(sentences[i.index(value)])) for value in i]
        ordered = sorted(tuples, key=lambda x: x[1], reverse=True)
        sims_.append(ordered)

    # Get the best sentences per topic according to similarity quantile
    best_sents = get_best_sents(sims_, quantile)
    
    # Combine the best sentences per topic to a summary within the maximum number of characters, maximizing similarity
    best_combo = get_best_combo(best_sents,sims_,num_of_char)
    
    # Create the summary
    summar = create_summary(best_combo,sentences)
    
    x.loc[index,'LSI summary'] = summar
    
    # ------------ Rouge score calculation between summaries and first 3 sentences ------- 
    # In the end, we did not use the Rouge score as a validation method for the number of sentences
    # because the method seemed cumbersome - especially regarding a missing method of normalization 
    
#    best_rouge = []
#    for n_top in range(1,4): 
#        num_topics = n_top
#        lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=num_topics)    
#        topics = lsi.show_topics(formatted=False)
#    
#        qvec = []
#        for topic in topics:
#            query = ' '.join(word[0] for word in topic[1])
#            qvec.append(query)
#    
#        vectorizer = TfidfVectorizer()
#        X = vectorizer.fit_transform(sentences)
#    
#        sims_ = []
#        for i in range(len(qvec)):
#            topic_sims = []
#            best_sims = []
#            a = vectorizer.transform([qvec[i]])
#            for j in range(X.shape[0]):
#                b = X[j]
#                topic_sims.append((j, cosine_similarity(a, b)[0][0], len(sentences[j])))
#            topic_sims = sorted(topic_sims, key=lambda x: x[1], reverse=True)
#            sims_.append(topic_sims)
#    
#        best_sents = get_best_sents(sims_, quantile)
#    
#        best_combo = get_best_combo(best_sents,sims_,num_of_char)
#    
#        summar = create_summary(best_combo,sentences)
#        
#       
#        
#        if len(best_rouge) == 0:
#            best_rouge.append([summar,rouge.get_scores(summar,(' '.join(sent for sent in sentences[0:3])))[0]['rouge-2']['f']])
#            
#        else: 
#            print(best_rouge[0][1])
#            if rouge.get_scores(summar,(' '.join(sent for sent in sentences)))[0]['rouge-l']['f']/len(best_combo)>best_rouge[0][1]:
#                best_rouge = [[summar,rouge.get_scores(summar,(' '.join(sent for sent in sentences[0:3])))[0]['rouge-l']['f']]]
#            
    
    
    #--------------- perform LDA -----------------
    corpus = [dictionary.doc2bow(sentence) for sentence in article_clean]
    NUM_OF_TOPICS = 3
    ldamodel = models.LdaModel(corpus, id2word=dictionary, num_topics=NUM_OF_TOPICS)
    topics = ldamodel.show_topics(formatted=False)
    
    # Order the topics by coherence score, which we use as a measure of relevance
    coherences = {}
    for n, topic in topics:
        topic = [word for word, _ in topic]
        cm = CoherenceModel(topics=[topic], corpus=corpus, coherence='u_mass', dictionary=dictionary)
        coherences[n] = abs(cm.get_coherence())

    coh_sorted = sorted(coherences.items(), key=operator.itemgetter(1))
    coh_indexes=[i[0] for i in coh_sorted]
    ordered_topics= [topics[i] for i in coh_indexes]
    ordered_topics = ordered_topics[0:3]
    
    # Create 3 topic summary
    summary = []
    sims_ = []
    for i in range(3):
        # Extract the topic and create topic query
        topic = ordered_topics[i][1]
        query = ' '.join(word[0] for word in topic)
        bow = dictionary.doc2bow(process_raw(query))
        q_vec = ldamodel[bow]
        # Get similarity between sentences and query vector
        sims = get_similarity(ldamodel, q_vec, corpus)
        sims = sorted(enumerate(sims), key=lambda item: -item[1])
        # Save similarity score and length for each sentence
        lengths = []
        for i in range(len(sims)):
            add_len = sims[i] + (len(sentences[sims[i][0]]),)
            lengths.append(add_len)
        sims_.append(lengths)

    # Get the best sentences per topic according to similarity quantile
    best_sents = get_best_sents(sims_, quantile)
    
    # Combine the best sentences per topic to a summary within the maximum number of characters, maximizing similarity
    best_combo = get_best_combo(best_sents,sims_,num_of_char)
    
    # Create the summary
    summar = create_summary(best_combo, sentences)

    # Save the LDA summary to the dataframe
    x.loc[index, 'LDA summary'] = summar

    #--------------- perform NMF ----------------- 
    vectorizer = CountVectorizer(tokenizer=process_raw, lowercase=True)
    X=vectorizer.fit_transform(sentences)
    df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names())

    # Tf-idf matrix
    TfidfVec = TfidfVectorizer(tokenizer=process_raw, lowercase=True)
    matrix = TfidfVec.fit_transform(sentences)
    tfidf_matrix = pd.DataFrame(matrix.toarray(), columns=TfidfVec.get_feature_names())
    
    num_topics = 3
    
    # Obtain a NMF model.
    modelnmf = NMF(n_components=num_topics, init='nndsvd')
    # Fit the model
    modelnmf.fit(tfidf_matrix) 
    
    # Extract the NMF topics
    nmf_topics=get_nmf_topics(modelnmf, vectorizer, 10)
    nmf_topics=nmf_topics.transpose().values.tolist()
    
    # Calculate coherence score of each topic
    coherences = {}
    for i in range(len(nmf_topics)):
        topic = [word for word in nmf_topics[i]]
        cm = CoherenceModel(topics=[topic], corpus=corpus, coherence='u_mass', dictionary=dictionary)
        coherences[i] = abs(cm.get_coherence())

    coh_sorted = sorted(coherences.items(), key=operator.itemgetter(1))
    coh_indexes=[i[0] for i in coh_sorted]
    ordered_topics= [nmf_topics[i] for i in coh_indexes]
    ordered_topics = ordered_topics[0:3]
    
    # Create query of topic
    qvec = []
    for topic in ordered_topics:
        query = ' '.join(word for word in topic)
        qvec.append(query)

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(sentences)

    # get similarities for each topic and the sentences
    sims_ = []
    for i in range(len(qvec)):
        topic_sims = []
        best_sims = []
        a = vectorizer.transform([qvec[i]])
        for j in range(X.shape[0]):
            b = X[j]
            topic_sims.append((j, cosine_similarity(a, b)[0][0], len(sentences[j])))
        topic_sims = sorted(topic_sims, key=lambda x: x[1], reverse=True)
        sims_.append(topic_sims)
        
    best_sents = get_best_sents(sims_, quantile)

    best_combo = get_best_combo(best_sents,sims_,num_of_char)

    summar = create_summary(best_combo,sentences)

    x.loc[index, 'NMF summary'] = summar
    
    #--------------- perform TextRank -----------------
    
    # Percentage of total sentences to include in summary
    perc=10/len(split_sentences(article))
    
    # Summarize
    tr_sum=summarize(article, ratio=perc)
    
    # Get list of <num_of_char long best sentences
    l=[]
    for i in split_sentences(tr_sum):
        if len(i) <= num_of_char:
            l.append(i)
    
    # Get best combo of max num_of_char
    fl=[]
    total_l=0
    for i in l:
        if total_l + len(i)<= num_of_char:
            total_l+=len(i)
            fl.append(i)
            
    #Concat final combo
    TR_Summary=' '.join(fl)
        
    x.loc[index, 'TR summary'] = TR_Summary
   

    #--------------- perform KMeans -----------------
    
    # Preprocessing
    list_of_sent = split_into_sentences(article)
    list_of_sent_of_word = [process_raw(sentence) for sentence in list_of_sent]
    flatten = [word for i in list_of_sent_of_word for word in i]   
    
    # sentence as vector representation of average of vector representation of words
    text2vec = []
    for sentence in list_of_sent_of_word:
        sentences2vec = []
        for word in sentence:
            try:
                sentences2vec.append(model[word])
            except:
                pass
        sentences2vec = np.average(sentences2vec, axis=0)
        text2vec.append(sentences2vec)
                
    
    # Centroid
    model_vec = []
    for word in flatten:
        try:
            model_vec.append(model[word])
        except:
            pass
        
    # Perform clustering
    n_clusters=3
    kmeans=KMeans(n_clusters=n_clusters)
    kmeans.fit(model_vec)
    centroids = kmeans.cluster_centers_
    cent_names = ['c1', 'c2', 'c3']
    
    # Find the cluster centroids
    for i in range(n_clusters):
        tpl = zip(cent_names, kmeans.cluster_centers_[i])
        
    df_similiarities = pd.DataFrame(columns = cent_names)

    # get similiarity between each sentence and the centroids 
    for i in range(n_clusters):
        a = 0
        for sentence in text2vec:
            df_similiarities.loc[a,cent_names[i]] = 1 - spatial.distance.cosine(kmeans.cluster_centers_[i], sentence)
            a+=1
    
    df_similiarities = df_similiarities.fillna(-1)    
    
    # Save similarities
    sims_ = []
    for i in range(len(df_similiarities.columns)):
        topic_sims = []
        best_sims = []
        for j in range(len(list_of_sent)):
            topic_sims.append((j, df_similiarities.iloc[j, i], len(list_of_sent[j])))
        topic_sims = sorted(topic_sims, key=lambda x: x[1], reverse=True)
        sims_.append(topic_sims)
        
    # Find best sentences, best combo and create summary
    best_sents = get_best_sents(sims_, quantile)

    best_combo = get_best_combo(best_sents,sims_,num_of_char)

    summar = create_summary(best_combo,list_of_sent)
    
    x.loc[index, 'kMeans summary'] = summar
    


#--------------- Performance evaluation of the models -----------------
# one row per title and each summary
df_sum = pd.DataFrame(columns = ['a', 'b', 'c'])
ix_of_df = [1, 3, 4, 5, 6, 7]
all_words = []

performance=pd.DataFrame(columns=['LSI', 'LDA', 'NMF', 'TextRank', 'kMeans'])

rr=0
for article in x.iterrows():
    rr+=1
    for index in ix_of_df:

        # process title and all summaries
        model_raw  = process_raw(article[1][index])
        # create word2vec for each processed model
        model_vec = []
        for word in model_raw:
            try:
                
                # create word2vec based on own trained model
                model_vec.append(model[word])
                all_words.append((word, model[word]))
                # if word is not in word2vec model, we skip it (it is not represented in the vector)
            except:
                pass
            # transform to array
            model_vec_rep = np.array(model_vec)   
            # take average of the word vectors for each method
            model_vec_rep_avg = np.average(model_vec_rep, axis=0)
            
        df_sum.loc[index,['a', 'b', 'c']] = [model_vec, model_vec_rep, model_vec_rep_avg]
        # Title to title comparison = 1!
        df_sum.loc[1,'sim'] = 1
        # Calculate cosine similarity between title vector and average vector of each method
        df_sum.loc[index,['sim']] = 1 - spatial.distance.cosine(df_sum.loc[1,'c'], model_vec_rep_avg)
        # We assign 0 as similarity for the title itself, so we don't pick title as the most similar
        df_sum.loc[1,'sim'] = 0
    app=pd.DataFrame(df_sum[1:]['sim'].values.reshape(1,5), columns=['LSI', 'LDA', 'NMF', 'TextRank', 'kMeans'])
    performance=performance.append(app, ignore_index=True)

# Twitterize the best performing summary by adding hashtags and mentions
link = url
tweet = x.iloc[0,df_sum['sim'].idxmax()]
tweet = twitterize(tweet).lstrip()
# Post the tweet on our Twitter Account
make_tweet(tweet,link)

# Plot the outcome of the word embedding
final_embed = []
for row in df_sum.iterrows():
    # final_embed stores the 4 average vectors
    final_embed.append(np.array(row[1]['c']))
    # This can plot us the 'positions' of the methods in the wordspace
final_embed = np.array(final_embed)

# for the word plot
labels_words = []
final_embed_all_words = []
for word, arr in all_words:
    final_embed_all_words.append(np.array(arr))
    labels_words.append(word)
final_embed_all_words = np.array(final_embed_all_words)

combine_embed = np.append(final_embed , final_embed_all_words, axis =0)

# Dimensionality reduction in order to show a 2D plot (strongly simplified)
tsne = TSNE(
    perplexity=2, n_components=2, init='pca', n_iter=5000, method='exact')


embedd = tsne.fit_transform(combine_embed)
# summaries & titles after dim reduction
low_dim_embs = embedd[0:6]
# words after dim reduction
low_dim_embs_words = embedd[6:]
labels = ['TITLE', 'LSI', 'LDA', 'NMF', 'TextRank', 'kMeans']
all_lab = [labels, labels_words]
plot_with_labels(low_dim_embs,low_dim_embs_words, all_lab,df_sum,   os.path.join(gettempdir(),'tsne.png'))
plt.show()                


#--------------- Overall performance evaluation and plots for the evaluation of the -----------------

#Average performance bar plot
#avg_compare=np.average(performance[['LSI', 'LDA', 'NMF', 'TextRank', 'kMeans']], axis=0)
#
#plt.bar(performance.columns[:5], avg_compare, color='#1da1f2')
#plt.title('Average Model Performance', fontsize=30, fontweight='bold', color='black')
#plt.xticks(fontsize=15)
#plt.yticks(fontsize=15)
#
##Winning counts bar plot
#best_counts=performance[['LSI', 'LDA', 'NMF', 'TextRank', 'kMeans']].idxmax(axis=1).value_counts(sort=False)
#
#plt.bar(best_counts.index, best_counts, color='#1da1f2')
#plt.title('Winning Model Counts', fontsize=30, fontweight='bold', color='black')
#plt.xticks(fontsize=15)
#plt.yticks(fontsize=15)
#
#
## Model Performance Boxplots
#performance.plot(kind='box',color='#1da1f2', boxprops=dict(linestyle='-', linewidth=2))
#plt.title('Model Performance Distribution', fontsize=30, fontweight='bold', color='#1da1f2')
#plt.xticks(fontsize=15)
#plt.yticks(fontsize=15)
#
#
##Best Summary Boxplots
#performance['Winner']=performance.idxmax(axis=1)
#for i in performance['Winner']:
#    performance['Winner Score']=performance[i]
#
#
#sns.set_style='whitegrid'
#sns.boxplot(x='Winner', y='Winner Score', data=performance[['Winner', 'Winner Score']], color='#1da1f2', boxprops=dict(linestyle='-', linewidth=2))
#plt.xlabel('')
#plt.ylabel('')
#plt.title('Best Summary distribution', fontsize=30, fontweight='bold', color='black')
#plt.xticks(fontsize=15)
#plt.yticks(fontsize=15)
#
##Bar chart Text Rank vs Our Models
#vs_TR=pd.DataFrame(columns=['TextRank', 'Our Score', 'Winner'])
#vs_TR['TextRank']=performance['TextRank']
#vs_TR['Our Score']=performance[['LSI', 'LDA', 'NMF', 'kMeans']].max(axis=1)
#vs_TR['Winner']=vs_TR.iloc[:,:2].idxmax(axis=1)
#vs_TR_counts=vs_TR.iloc[:,-1].value_counts()
#
#plt.bar(vs_TR_counts.index, vs_TR_counts, color='#1da1f2')
#plt.title('Text Rank vs our model wins', fontsize=30, fontweight='bold', color='black')
#plt.ylabel('Winning model counts',fontsize=15)
#plt.xticks(fontsize=15)
#plt.yticks(fontsize=15)
#
#
#
