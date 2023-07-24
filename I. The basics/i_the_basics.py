"""
Date created: Feb 9 2023
Last updated: Feb 9 2023

Lesson I of DEAL Python series: "the basics"
"""

# import necessary packages
from lyricsgenius import Genius
import re   # this is Python's built-in "regular expressions" pacakge

########################################################################################################################
# variables and data types
########################################################################################################################

# defining a variable is easy (just use a single equals sign)
x = 1
print('x =', x)

# some variable naming rules: no spaces, the name can't start with a number, and the name shouldn't be the same as
# a built-in python object

# basic data types (classes) to know: (1) booleans (2) numbers, (3) strings, (4) lists, (5) dictionaries

# (1) booleans
#####################################
# booleans or "bools" can take only two values: True or False
my_bool = True
print(my_bool)
print(type(my_bool))   # use the type function to get the type of any python object

# booleans are often used in control flow statements as we'll see below

# often used to turn debug/print statements on or off (as we'll see below)
debug = True

# (2) numbers
#####################################
# numbers can be 'int' types (integers) or 'float' types
y = 2
print('y =', y, 'is type:', type(y))
z = y / x   # division ALWAYS returns a float
print('z =', z, 'is type:', type(z))

# basic math operations are built in
# use '+' to add
# use '-' to subtract
# use '*' to multiply
# use '**' to calculate powers
my_num = x + y + 14 - 2**2  # addition, subtraction, and a power
print(my_num)

# while a single equals sign is used for defining variables, a double equals sign is used to test for equality,
# and will return a boolean
print(my_num == 13)
print(my_num == 14)

# (3) strings
#####################################

# we have already used strings in our print statements above
mystring = "Taylor Swift's favorite color is"
print(mystring)

# you can use single quotes or double quotes; Python treats them the same
# NOTE: if you use single quotes and have a single quote INSIDE your string, you can use \ to "escape" the quote
mystring_2 = 'Taylor Swift\'s favorite color is'
print(mystring_2)

# you can "add" strings together using '+'
red = 'red'
print(mystring + ' ' + red)

# strings are "subscriptable"
# NOTE: Python indexes always start at 0 (not 1)!
print(mystring[0])   # character in position 0 (first character)
print(mystring[5])   # character in position 5
print(mystring[-1])   # character in the last position (negative index starts count from the right)

# you can take "slices" from strings
new_string = mystring[0:12]   # take the first 12 characters of the string
print(new_string)
# you can add a third argument in the brackets to specify the "step" by which characters are taken from the string
new_string_2 = mystring[::-1]   # reverse the string
print(new_string_2)
new_string_3 = mystring[0:12:2]   # take EVERY SECOND character of the first 12 characters
print(new_string_3)

# strings are IMMUTABLE (you cannot change individual characters in a string)
try:
    mystring[5] = 'k'
except TypeError:
    print('Remember, strings aren\'t subscriptable!')

# use the len function to get the length of a string
print('the length of mystring is:', len(mystring))

# strings have a number of built-in functions, which you can read more about in Python's documentation
mystring_lower = mystring.lower()   # e.g., convert all characters to lower-case
print(mystring_lower)

# (4) lists
#####################################

# lists are written comma-separated values (items) between square brackets
squares = [1, 4, 9, 16, 25]
print(squares)
colors = ['red', 'maroon', 'lavender']
print(colors)

# lists might contain items of different types, but usually the items all have the same type
numbers = [1, 'two', 3, 'four']
print(numbers)

# lists are add-able and subscriptable like strings
squares_and_colors = squares + colors
print(squares_and_colors)
print(squares[1:4])

# lists are nestable (you can have lists within lists, to arbitrary levels)
apple = ['a', 'p', 'p', 'l', 'e']
orange = ['o', 'r', 'a', 'n', 'g', 'e']
fruits = [apple, orange]
print(fruits)

# when lists are nested, you must use indices for each layer to get items in sub-lists
print(fruits[0])
print(fruits[0][2])

# unlike strings, lists are MUTABLE (you can replace individual items in the list using indices)
print(numbers)
numbers[1] = 2
print(numbers)

# (5) dictionaries
#####################################

# dictionaries allow you to associate a set of unique "keys" with a set of "values"
# in a language dictionary, the keys are words and the values are the definitions
# in a python dictionary, the keys can be any immutable objects, and the values can be any python objects

# a simple dictionary of Taylor Swift's first three albums
# the keys are the order in which they came out, and the values are the album names
taylors_albums = {1: 'Taylor Swift', 2: 'Fearless', 3: 'Speak Now'}

# to "call" specific items from a dictionary, use the keys
print('Taylor Swift\'s second album was:', taylors_albums[2])

# dictionaries are mutable
taylors_albums[3] = 'red'
print(taylors_albums)

########################################################################################################################
# Interlude: using an external package and API token to read in Taylor Swift lyrics
########################################################################################################################

genius_token = 'xpJzuajigwd_isrImhCQKWHLpB0dOJz-379V_oAOv9fEuNFsiqZMjcwG9P300zVr'
genius = Genius(genius_token)

# Read in the lyrics for (1) Taylor Swift's 25 most popular songs on Genius, and (2) All of the songs from her album
# "Folklore" and create dictionaries for each set of lyrics mapping the song title to the lyrics
#####################################################################################################################

# collect Taylor Swift's 25 most popular songs, with lyrics, from Genius
tswift = genius.search_artist('Taylor Swift', max_songs=25, sort='popularity')

# use a for loop to create a dictionary mapping song names to their lyrics
# for loops iterate over every item in a sequence (e.g., in a list, string, or dictionary)
tswift_lyrics = {}   # first we initialize an empty dictionary
for s in tswift.songs:
    title = s.title.replace('\u200b', '')   # fix unicode formatting issue with all-lowercase titles
    lyrics = s.lyrics.replace('\u205f', ' ') # fix unicode formatting issue
    lyrics = lyrics.replace('\u2005', ' ')
    lyrics = lyrics.replace('\u200b', ' ')
    tswift_lyrics[title] = lyrics

# now lets get all the songs from her 'folklore' album
folklore = genius.search_album("Folklore", "Taylor Swift")

# again, use a for loop to create a dictionary
folklore_lyrics = {}
for s in folklore.tracks:
    title = s.song.title.replace('\u200b', '')   # fix unicode formatting issue with all-lowercase titles
    lyrics = s.song.lyrics.replace('\u205f', ' ') # fix unicode formatting issue
    lyrics = lyrics.replace('\u2005', ' ')
    lyrics = lyrics.replace('\u200b', ' ')
    folklore_lyrics[title] = lyrics

# now, we have a dictionary called tswift_lyrics, which contains lyrics for her top 25 most popular songs, and a
# dictionary called folklore_lyrics, which contains lyrics for all of the songs on the album folklore

# example: choose a song title from Folklore to get the lyrics from our new dictionary (the title is the key)
print('the lyrics to cardigan are:')
print(folklore_lyrics['cardigan'])

# save the 'cardigan' lyrics as their own object (a long string); we'll use this later
cardigan_lyrics = folklore_lyrics['cardigan']
# calling .split on a string creates a list of strings, split by the specified character
cardigan_lyrics = cardigan_lyrics.split("\n")

########################################################################################################################
# Control flow statements
########################################################################################################################

# for loops
#####################################
# for loops allows us to iterate through a sequence (e.g., a string or list)

# use a for loop to print all the track names on Folklore
# since the track names are the KEYS of the dictionary, we'll use the '.keys()' argument at the end of the dictionary
# name to get a list of the keys, which we'll iterate through
for track in folklore_lyrics.keys():
    print(track)

# use a for loop + range to iterate through a sequence based on indices
# print the first four lines of the cardigan lyrics
for i in range(4):
    print(cardigan_lyrics[i])

# if statements
#####################################
# if statements allow us to execute a command ONLY IF certain conditions are met

# add an if statement to our for loop above to print ONLY the songs from Folklore that are also in the top 25 most
# popular songs on Genius lyrics
for track in folklore_lyrics.keys():
    if track in tswift_lyrics.keys():
        print(track)

# we can also compute the percent of songs from Folklore that are in the top 25 most popular on Genius lyrics
count = 0   # initiate a count variable to keep track of the songs appearing in the top 25
for track in folklore_lyrics.keys():
    if track in tswift_lyrics.keys():
        count += 1   # since this comes after the 'if' statement, the count will only increase when the condition is met
        if debug:   # only print the track if we set debug == True
            print(track)

print('the percent of songs from Folklore that are also in Taylor\'s top 25 most popular are:', count / len(folklore_lyrics))

# we can use an "else" line to make something else happen if the condition in the if statement isn't satisfied
for track in folklore_lyrics.keys():
    if track in tswift_lyrics.keys():
        print(track, 'is one of the top 25 most popular')
    else:
        print(track, 'is not very popular')

# we can use "elif" to specify multiple conditions to check
for track in folklore_lyrics.keys():
    if track in tswift_lyrics.keys():
        print(track)
    elif track == 'hoax':
        print(track, 'is really pretty')
    else:
        print(track, 'is not very popular')

# while loops
#####################################
# while statements allow you to continue executing command UNTIL a condition is satisfied

# use a while statement to print the lines from cardigan UNTIL the word cardigan is actually used
cardigan_not_used = True   # initiate our condition (the word "cardigan" hasn't shown up yet)
line = 1   # start with line 1 of the lyrics
while cardigan_not_used:
    print(cardigan_lyrics[line])
    line += 1   # update the line number
    cardigan_not_used = ('cardigan' not in cardigan_lyrics[line])   # update our condition

########################################################################################################################
# Functions
########################################################################################################################

# Python allows you to define functions, which are useful for repetitive processes


def power(x, n):
    """
    :param x: the number you want to raise to a given power
    :param n: the power to raise x to (limit to positive integers for simplicty here)
    :return: x^n
    """
    y = x

    for i in range(n-1):
        y *= x

    return y


print('4 cubed is', power(4,3))


########################################################################################################################
# Example: count the number of times colors appear in the lyrics of Taylor's most popular songs
########################################################################################################################


# Define a function to count the number of times a word appears in a string
def count_instances(word, corpus):
    """
    :param word: a word you want to count the instances of (should be string type)
    :param corpus: the document you want to count instances in (should also be string type)
    :return: how many times the word appears in your corpus (an integer)
    """
    word_count = 0
    corpus = corpus.lower() # use .lower() arguments to avoid case sensitivity
    word = word.lower()

    # naive word count
    # word_count += corpus.count(word)

    # count only if the word appears by itself (not nested within another word)
    word_count = len(re.findall(r'\b' + word + r'\b', corpus))

    return word_count


# let's use our function to see how many times the word 'blue' appears in the song 'hoax'
print('blue appears in the song "hoax"', count_instances('blue', folklore_lyrics['hoax']), 'times')


colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'violet', 'lavender', 'pink', 'white', 'silver',
          'gold', 'black', 'brown', 'maroon', 'scarlet', 'gray']

colors_count_top_25 = {}

# count the number of times each color appears in Taylor's 25 most popular songs
for c in colors:
    for s in tswift_lyrics.keys():
        color_count = count_instances(c, tswift_lyrics[s])
        if debug:
            print(c, 'appears in the song', s, color_count, 'times')
        if c in colors_count_top_25.keys():
            colors_count_top_25[c] += color_count
        else:
            colors_count_top_25[c] = color_count

# mini visualization
colors_ranked = []
for v in set(sorted(list(colors_count_top_25.values()))):
    colors_ranked += [i for i in colors_count_top_25 if colors_count_top_25[i] == v]
print('The colors appearing most often in Taylor\'s top 25 songs (ascending):')
for c in colors_ranked:
    print(c, ':', ' '*(10-len(c)),  '|'*colors_count_top_25[c])

# count the number of times each color appears in the Folklore album
colors_count_folklore = {}

for c in colors:
    for s in folklore_lyrics.keys():
        color_count = count_instances(c, folklore_lyrics[s])
        print(c, 'appears in the song', s, color_count, 'times')
        if c in colors_count_folklore.keys():
            colors_count_folklore[c] += color_count
        else:
            colors_count_folklore[c] = color_count

# mini visualization
colors_ranked = []
for v in set(sorted(list(colors_count_folklore.values()))):
    colors_ranked += [i for i in colors_count_folklore if colors_count_folklore[i] == v]
print('The colors appearing most often in Folklore (ascending):')
for c in colors_ranked:
    print(c, ':', ' '*(10-len(c)),  '|'*colors_count_folklore[c])





