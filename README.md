# WDM-Ex.NO-8-TEXT-CLASSIFICATION
Aim:
To perform text classification for movie reviews using python.
Description:
Steps required to create a text classification model in Python:
1.	Importing Libraries
2.	Importing The dataset
3.	Text Preprocessing
4.	Converting Text to Numbers
5.	Training and Test Sets
6.	Training Text Classification Model and Predicting Sentiment
7.	Evaluating the Model
8.	Saving and Loading the Model

   
**Program:**
```
import numpy as np import re
import nltk nltk.download('wordnet')
from sklearn.ensemble import RandomForestClassifier from sklearn.datasets import load_files
from nltk.stem import WordNetLemmatizer nltk.download('stopwords')
import pickle
from nltk.corpus import stopwords
movie_data = load_files(r"/content/sample_data/txt_sentoken")
X, y = movie_data.data, movie_data.target documents = []
stemmer = WordNetLemmatizer() for sen in range(0, len(X)):
# Remove all the special characters document = re.sub(r'\W', ' ', str(X[sen])) # remove all single characters
 
document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document) # Remove single characters from the start document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) # Substituting multiple spaces with single space document = re.sub(r'\s+', ' ', document, flags=re.I) # Removing prefixed 'b'
document = re.sub(r'^b\s+', '', document) # Converting to Lowercase
document = document.lower() # Lemmatization
document = document.split()
document = [stemmer.lemmatize(word) for word in document] document = ' '.join(document)
documents.append(document)
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english')) X = vectorizer.fit_transform(documents).toarray()
from sklearn.feature_extraction.text import TfidfTransformer tfidfconverter = TfidfTransformer()
X = tfidfconverter.fit_transform(X).toarray()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) classifier = RandomForestClassifier(n_estimators=1000, random_state=0) classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred)) print(accuracy_score(y_test, y_pred))
 ```
**dataset1.txt:**
plot : two teen couples go to a church party , drink and then drive . they get into an accident .
one of the guys dies , but his girlfriend continues to see him in her life , and has nightmares . what's the deal ?
watch the movie and " sorta " find out . . .
critique : a mind-fuck movie for the teen generation that touches on a very cool idea , but presents it in a very bad package .
which is what makes this review an even harder one to write , since i generally applaud films which attempt to break the mold , mess with your head and such ( lost highway & memento ) , but there are good and bad ways of making all types of films , and these folks just didn't snag this one correctly .
they seem to have taken this pretty neat concept , but executed it terribly . so what are the problems with the movie ?
well , its main problem is that it's simply too jumbled .
it starts off " normal " but then downshifts into this " fantasy " world in which you , as an audience member , have no idea what's going on .
there are dreams , there are characters coming back from the dead , there are others who look like the dead , there are strange apparitions , there are disappearances , there are a looooot of chase scenes , there are tons of weird things that happen , and most of it is simply not explained .
now i personally don't mind trying to unravel a film every now and then , but when all it does is give me the same clue over and over again , i get kind of fed up after a while , which is this film's biggest problem .
it's obviously got this big secret to hide , but it seems to want to hide it completely until its final five minutes . and do they make things entertaining , thrilling or even engaging , in the meantime ?
not really .
the sad part is that the arrow and i both dig on flicks like this , so we actually figured most of it out by the half-way point , so all of the strangeness after that did start to make a little bit of sense , but it still didn't the make the film all that more entertaining .
i guess the bottom line with movies like this is that you should always make sure that the audience is " into it " even before they are given the secret password to enter your world of understanding .
i mean , showing melissa sagemiller running away from visions for about 20 minutes throughout the movie is just plain lazy ! !
okay , we get it . . . there are people chasing her and we don't know who they are . do we really need to see it over and over again ?
how about giving us different scenes offering further insight into all of the strangeness going down in the movie ? apparently , the studio took this film away from its director and chopped it up themselves , and it shows .
 
there might've been a pretty decent teen mind-fuck movie in here somewhere , but i guess " the suits " decided that turning it into a music video with little edge , would make more sense .
the actors are pretty good for the most part , although wes bentley just seemed to be playing the exact same character that he did in american beauty , only in a new neighborhood .
but my biggest kudos go out to sagemiller , who holds her own throughout the entire film , and actually has you feeling her character's unraveling .
overall , the film doesn't stick because it doesn't entertain , it's confusing , it rarely excites and it feels pretty redundant for most of its runtime , despite a pretty cool ending and explanation to all of the craziness that came before it .
oh , and by the way , this is not a horror or teen slasher flick . . . it's
just packaged to look that way because someone is apparently assuming that the genre is still hot with the kids . it also wrapped production two years ago and has been sitting on the shelves ever since .
whatever . . . skip it !
where's joblo coming from ?
a nightmare of elm street 3 ( 7/10 ) - blair witch 2 ( 7/10 ) - the crow ( 9/10 ) - the crow : salvation ( 4/10 ) - lost highway ( 10/10 ) - memento ( 10/10 ) - the others ( 9/10 ) - stir of echoes ( 8/10 )

**Output:**
<img width="733" height="467" alt="image" src="https://github.com/user-attachments/assets/a8f40236-31c0-4e1f-9ee5-ed68ff3bfd8e" />

**Result:**
Thus, the text classification has been performed for predicting the movie reviews using python done successfully.


