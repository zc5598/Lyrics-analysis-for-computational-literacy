import nltk
nltk.download()

from nltk.corpus import stopwords
nltk.download('stopwords')

nltkstopwords = set(stopwords.words('english'))

extraword = nltkstopwords.union({'oh','ooh','mm','hmm','ah','wooo','gon','na'})

print(extraword)

from nltk.tokenize import word_tokenize

lyrics = """A scrub is a guy that thinks he's fly
And is also known as a busta (Busta)
Always talkin' 'bout what he wants
And just sits on his broke ass, so

No, I don't want your number (Uh, uh-uh)
No, I don't wanna give you mine and
No, I don't wanna meet you nowhere (No)
No, don't want none of your time and (Uh)

No, I don't want no scrub (Uh)
A scrub is a guy that can't get no love from me
Hangin' out the passenger side of his best friend's ride
Trying to holla at me (Woo)
I don't want no scrub
A scrub is a guy that can't get no love from me (Uh-uh, no love)
Hangin' out the passenger side of his best friend's ride (Uh-uh, uh-uh, uh-uh)
Trying to holla at me

Well, a scrub checkin' me, but his game is kinda weak
And I know that he cannot approach me
'Cause I'm lookin' like class and he's lookin' like trash
Can't get with a deadbeat ass, so (Yeah)

No, I don't want your number
No, I don't wanna give you mine and (Said no)
No, I don't wanna meet you nowhere
No, don't want none of your time (Check it)

No, I don't want no scrub
A scrub is a guy that can't get no love from me (Uh-huh, come on)
Hangin' out the passenger side of his best friend's ride
Trying to holla at me (Hah)
I don't want no scrub
A scrub is a guy that can't get no love from me (Uh-uh, no love)
Hangin' out the passenger side of his best friend's ride
Trying to holla at me

If you don't have a car and you're walkin'
Oh, yes, son, I'm talkin' to you
If you live at home with your mama
Oh, yes, son, I'm talkin' to you (Baby)
If you have a shorty that you don't show love
Oh, yes, son (Yeah), I'm talkin' to you (Yeah)
Wanna get with me with no money, oh no
I don't want no
No (No scrubs), scrub (No love)
No (No scrubs), scrub (No, no, no love)
No (Uh-uh, uh-uh, scrub), scrub (Uh-uh, uh-uh, no love, no, no, no, no)
No (Uh-uh, uh-uh), scrub (No, no, uh)

No, I don't want no scrub
A scrub is a guy that can't get no love from me
Hangin' out the passenger side of his best friend's ride (Yeah)
Trying to holla at me
I don't want no scrub (No)
A scrub is a guy that can't get no love from me
Hangin' out the passenger side of his best friend's ride (Yeah)
Trying to holla at me

See, if you can't spatially expand my horizons (Horizons)
Then that leaves you in a class with scrubs, never risin' (Risin')
I don't find it surprisin' if you don't have the G's
To please me and bounce from here to the coast of overseas
So, let me give you somethin' to think about (Think about)
Inundate your mind with intentions to turn you out (Turn you out)
Can't forget the focus on the picture in front of me (In front of me)
You as clear as DVD on digital TV screens (Digital TV screens)
Satisfy my appetite with somethin' spectacular (Spectacular)
Check your vernacular, and then I'll get back to ya (Then I'll get back to ya)
With diamond-like precision, insatiable is what I envision
Can't detect acquisition from your friend's Expedition
Mister Big Willy
If you really wanna know, ask Chilli
Could I be a silly ho? Not really
T-Boz and all my señoritas are steppin' on your FILAs
But you don't hear me, though


No, I don't want no scrub (No)
A scrub is a guy that can't get no love from me (Uh-uh, uh-uh, uh)
Hangin' out the passenger side of his best friend's ride (Uh, yeah)
Trying to holla at me
I don't want no scrub (No scrub)
A scrub is a guy that can't get no love from me (No love)
Hangin' out the passenger side of his best friend's ride
Trying to holla at me
"""


tokens = word_tokenize(lyrics)

filtered = [w for w in tokens if not w.lower() in extraword]

filtered = []

for w in tokens:
	if w not in extraword:
		filtered.append(w)

print(tokens)
print(filtered)

import string

filtered = [w.lower() for w in tokens if w.isalpha() and w.lower() not in extraword]

print("Filtered Words:")
print(filtered)

cleaned_text = " ".join(filtered)

print("Cleaned text:")
print(cleaned_text)


