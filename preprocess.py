import nltk
nltk.download()

from nltk.corpus import stopwords

nltk.download('stopwords')
print(stopwords.words('english'))

from nltk.tokenize import word_tokenize

example_sent = """If I should stay
I would only be in your way
So I'll go, but I know
I'll think of you every step of the way

And I
Will always love you
I will always love you

You
My darling, you
Mm, mm

Bittersweet memories
That is all I'm taking with me
So goodbye, please, don't cry
We both know I'm not what you, you need

And I
Will always love you
I will always love you
You

I hope life treats you kind
And I hope you have all you've dreamed of
And I wish to you joy and happiness
But above all this, I wish you love

And I
Will always love you
I will always love you
I will always love you
I will always love you
I will always love you, I
I will always love you, you

Darling, I love you
Ooh, I'll always
I'll always love you"""

stop_words = set(stopwords.words('english'))

word_tokens = word_tokenize(example_sent)

filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]

filtered_sentence = []

for w in word_tokens:
	if w not in stop_words:
		filtered_sentence.append(w)

print(word_tokens)
print(filtered_sentence)

import string

filtered_sentence = [w.lower() for w in word_tokens if w.isalpha() and w.lower() not in stop_words]

print("Filtered Words:")
print(filtered_sentence)

cleaned_text = " ".join(filtered_sentence)

print("Cleaned Text:")
print(cleaned_text)


