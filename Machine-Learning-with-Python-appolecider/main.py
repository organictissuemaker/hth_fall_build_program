from sklearn import tree
from sklearn.feature_extraction.text import CountVectorizer  #text to numbers
import matplotlib.pyplot as plt

positive_texts = [
    "we love you",
    "they love us",
    "you are good", 
    "he is good",
    "they love mary"
]

negative_texts = [
    "we hate you", 
    "they hate us",
    "you are bad", 
    "he is bad", 
    "we hate mary"
    
]

test_texts = [
    "people love cats and dogs equally", 
    "they are good",
    "why do you hate mary", 
    "they are almost always good", 
    "we are very bad",
    "they dislike sports"
]
'''
Here are three simple datasets. The first one contains five positive sentences; the second one contains five negative sentences; and the last contains a mix of both.

We can easily see which sentences are positive and which are negative, but can we teach a computer to do this?

We'll use the two lists of positive and negative sentences to train a model with Python. We will give examples to the computer that are already labelled as positive or negative. The computer will compute how to find the difference, and then we'll test it with our mixed sentences. The computer will guess whether each example is positive or negative.
'''

# Vectorization Example: Let's say we have two sample lists:

sample_sentences = [["nice pizza is nice"], ["what is pizza"]]

# First we go through can find all the unique words and give them a number. Those words would be "nice", "pizza", "is", and "what"

unique_words = {"nice": 0, "pizza": 1, "is": 2, "what": 3}

# Next we go through our sentences from left to right and create vectors based on how many times each of the unique words in our vocabulary appears in that particular sentence

vectors = [[2, 1, 1, 0], [0, 1, 1, 1]]

# this is showing us that in our first sentence, the word nice appears twice, pizza appears once, is appears once and what does not appear at all. In our second sentence, nice does not appear, while all the other three appear just once

training_texts = negative_texts + positive_texts

training_labels = ["negative"] * len(negative_texts) + ["positive"
                                                        ] * len(positive_texts)

sentence = (training_texts[3], training_labels[3])
'''
Here we are creating the labels that catagorize each sentence as either positive our negative. Here is what our two new variables look like:

training_text = ['we hate you', 'they hate us', 'you are bad', 'he is bad', 'we hate mary', 'we love you', 'they love us', 'you are good', 'he is good', 'they love mary']

training_labels = ['negative', 'negative', 'negative', 'negative', 'negative', 'positive', 'positive', 'positive', 'positive', 'positive']

if we wanted to grab each sentence and its appropriate label we could use the same index position in each variable:

first_sentence = (training_text[0] training_labels[0])
'''

vectorizer = CountVectorizer()

vectorizer.fit(training_texts)

unique_words = vectorizer.vocabulary_
#print(unique_words)
'''
Here we can already see how Python makes this process much easier. Instead of having to go through and hand pick all our unique words, the scikit-learn vectorizer handles that for us. You can use a print() statement to see your unique words. It should look something like this:

uniqe_words = {
    'are': 0,
    'bad': 1,
    'good': 2
    'hate': 3, 
    'he': 4,
    'is': 5,
    'love':6,
    'mary': 7, 
    'they': 8,
    'us': 9,  
    'we': 10,
    'you': 11}

1 = 'we hate you'
vector = [1, 1, 1, 0, 0, 0 ,0, 0, 0, 0, 0, 0]

2 = 'they love us'
vector = [0, 0, 0, 1, 1, 0 ,0, 0, 0, 0, 1, 0]

The only difference between this and our earlier example is that the numbers we given based on the words alphabetically order.
'''

training_vectors = vectorizer.transform(training_texts)
#print(training_vectors.toarray())

testing_vectors = vectorizer.transform(test_texts)
classifier = tree.DecisionTreeClassifier()

classifier.fit(training_vectors, training_labels)

predictions = classifier.predict(testing_vectors)
print("These are the classifier predictions: \n {}".format(predictions))

'''
When creating our classifier, we finally make use of our training_labels variable, so that our algorithm knows what sort of outputs it should be giving. Once we print our predictions we should see something like:

predictions = ['positive' 'positive' 'negative' 'positive' 'negative']

We would then compare there predictions to our original test sentences:

test_texts = ["they love mary", "they are good", "why do you hate mary", "they are almost always good", "we are very bad"]

It looks like the computer has learned well! The words "bad" and "hate" appear only in the negative texts and the words "good" and "love", only in the positive ones. Other words like "they", "mary", "you" and "we" appear in both. A well trained model will have learned to ignore the words that appear in both, and focus on "good", "bad", "love" and "hate".
'''

fig = plt.figure(figsize=(5,5))

tree.plot_tree(classifier,feature_names = vectorizer.get_feature_names_out(), rounded = True, filled = True)

fig.savefig('tree.png')

'''
One feature of matplotlib we haven't talked much about is the ability to save any visualizations you create! You can download visualizations as a regular image or a gif, depending on which is more appropriate. The savefig() method makes this quick and simple. You should see a file called 'tree.png' show up near your "main.py" in the file explorer. If you open it, you can see your tree graph.
'''
# Compare this manually built classifier to your new machine learning tree algorithm

predictions2 = []

def manual_classify(text):
    if "hate" in text:
        return "negative"
    if "bad" in text:
        return "negative"
    return "positive"

predictions = []

for text in test_texts:
    prediction = manual_classify(text)
    predictions2.append(prediction)
  
print("These are the classifier predictions: \n {}".format(predictions2))