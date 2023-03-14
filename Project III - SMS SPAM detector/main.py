import pandas as pd
import re
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Read CSV file
sms_data = pd.read_csv("spam.csv", encoding='latin-1')

# We only need 1st and 2nd column in the original data for consideration
cols = sms_data.columns[:2]
data = sms_data[cols]
data = data.rename(columns={"v1": "Value", "v2": "Text"})

# Finding number of punctuations in the text column
data["Punctuations"] = data["Text"].apply(lambda x: len(re.findall(r"[^\w+&&^\s]", x)))

# Finding if any phone number is present in the text [1: yes, 0: no]
data["Phonenumbers"] = data["Text"].apply(lambda x: len(re.findall(r"[0-9]{10}", x)))

# Finding if any link is present in the text [1: yes, 0: no]
is_link = lambda x: 1 if re.search(r"https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+", x) != None else 0
data["Links"] = data["Text"].apply(is_link)

# Count number of uppercase words in text
count_upper = lambda x: list(map(str.isupper, x.split())).count(True)
data["Uppercase"] = data["Text"].apply(count_upper)

# Creating instance for TfidfVectorizer which calulates rarity of words in the corpus
tf_idf = TfidfVectorizer(stop_words="english", strip_accents='ascii', max_features=500)
tf_idf_matrix = tf_idf.fit_transform(data["Text"])
data_extra_features = pd.concat([data, pd.DataFrame(tf_idf_matrix.toarray(), columns=tf_idf.get_feature_names_out())],
                                axis=1)

# Dropping 'Value' & 'Text' columns and just feeding the classifier with rest of the features. Also, splitting the data.
X = data_extra_features
features = X.columns.drop(["Value", "Text"])
target = ["Value"]
X_train, X_test, y_train, y_test = train_test_split(X[features], X[target], random_state=42)

# Training the Classifier
dt = DecisionTreeClassifier(min_samples_split=40)
dt.fit(X_train, y_train)
predicted_val = dt.predict(X_test)
print(accuracy_score(y_train, dt.predict(X_train)))  # 0.9885139985642498
print(accuracy_score(y_test, predicted_val))  # 0.9720028715003589
print(f1_score(y_test, predicted_val, pos_label='spam'))  # 0.8959999999999999