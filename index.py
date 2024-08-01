import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from flask import Flask, request, render_template
import chardet
import imaplib
import email
from email.header import decode_header

application = Flask(__name__)
app = application
list = []
# Load spam detection model
with open('Data/spam.csv', 'rb') as f:
    result = chardet.detect(f.read())
    encoding = result['encoding']

df = pd.read_csv('Data/spam.csv', encoding=encoding)
df = df[['v1', 'v2']]

df['result'] = df['v1']
df['message'] = df['v2']

df = df.drop(labels=['v1', 'v2'], axis=1)

x = np.array(df['message'])
y = np.array(df['result'])

cv = CountVectorizer()
X = cv.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)

clf = MultinomialNB()
clf.fit(x_train, y_train)

def spam(text):
    data = cv.transform([text]).toarray()
    res = clf.predict(data)
    return res

## For email processing
df1 = pd.read_csv('Data/emails.csv')

df1['result'] = df1['spam']
df1['message'] = df1['text']

df1 = df1.drop(labels=['spam', 'text'], axis=1)

x1 = np.array(df1['message'])
y1 = np.array(df1['result'])

cv1 = CountVectorizer()
X1 = cv1.fit_transform(x1)

x_train1, x_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=.3, random_state=42)
clf1 = MultinomialNB()
clf1.fit(x_train1, y_train1)

def email125(subject):
    data1 = cv1.transform([subject]).toarray()
    res1 = clf1.predict(data1)
    return res1

def count_spam_emails(username, password):
    spam_count = 0
    try:
        mail = imaplib.IMAP4_SSL('imap.gmail.com')
        mail.login(username, password)
        mail.select('inbox')

        result, data = mail.search(None, 'ALL')
        email_ids = data[0].split()

        for email_id in email_ids:
            result, msg_data = mail.fetch(email_id, '(RFC822)')
            msg = email.message_from_bytes(msg_data[0][1])

            subject, encoding = decode_header(msg['Subject'])[0]
            if isinstance(subject, bytes):
                subject = subject.decode(encoding if encoding else 'utf-8')

            if email125(subject) == 1:
                list.append(subject)
                spam_count += 1

        mail.logout()
    except Exception as e:
        print(f"An error occurred: {e}")

    return spam_count

@app.route("/", methods=['GET'])
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form.get('text')
    result = spam(text)
    result_label = "Spam" if result[0] == 'spam' else "Not Spam"
  
    return render_template('index.html', message=result_label)

@app.route("/login", methods=['GET'])
def login():
    return render_template("login.html")

@app.route('/spam1', methods=['POST'])
def spam1():
    username = request.form.get('username')
    password = request.form.get('password')

    spam_count = count_spam_emails(username, password)
    result_label1 = f"Number of Spam Emails: {spam_count}"
    
    return render_template('login.html', message=result_label1)



@app.route('/list')
def list_view():
    # Example list to be passed to the template
    my_list =list
    return render_template('list.html', items=my_list)


if __name__ == "__main__":
    app.run(host="0.0.0.0")

##APP PASSWORD     gdck crxe focx snkw