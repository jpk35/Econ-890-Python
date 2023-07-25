"""
24 July 2023
Econ 890 session IV: web scraping

required packages:
--beautifulsoup4
--pandas
--requests
--selenium
"""

# toggle debug print statements on or off
debug = True

########################################################################################################################
# PART 1: Scraping static webpages (Web 1.0) with requests and beautifulsoup

# Example: collect the most recent earnings call transcripts from Motley Fool
########################################################################################################################

# import necessary packages
import requests
from bs4 import BeautifulSoup
import re   # to construct regular expressions patterns for searches
import pandas as pd   # to collect results in a dataframe

# the basics: request and parse the html of the Motley Fool earnings call transcripts page
###########################################################################################

# get the html of the website you'd like to scrape data from using requests
url = 'https://www.fool.com/earnings-call-transcripts/'

# use beautifulsoup to parse the html
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

# let's see what the html source code of this page looks like
# (beautifulsoup "prettifies" it for us!)
print(soup.prettify())

# we can also quickly grab and inspect all of the text on the page
print(soup.get_text())

# some basic html tags:
# (note: html structures generally have lots of nesting, so tags can and usually do live within other tags)
# <a> = links
# <button> = buttons
# <h1>, <h2>, etc. = headers
# <ol> = ordered lists, <ul> = unordered lists, <menu> = menu lists
# -- <li> tags list items within each of these lists types
# <img> = image
# <input> = place for user input (e.g., search box)
# <table> = table
# <p> = paragraph (used to format text)
# <div> = miscellaneous, text often lives here, too

# e.g., let's find all the links on this page
all_links = soup.find_all('a')

# let's see what the first few links look like:
for i in range(3):
    print(all_links[i])

# you can also find elements using:
# --css selector
# --element attributes (class name, id, href text, text within element, etc.)

# e.g., find element by css selector:
css_selector = 'body > div.main-container > div.page-grid-container > section > header > div > a'
logo = soup.select(css_selector)
print(logo)

# find the same element by class name
logo_2 = soup.find(class_='flex h-full logo')
print(logo_2)

# collect all the transcript links
############################################################################

# combine search strategies to find all links whose href contains the text "earnings-call-transcript"
transcript_links = soup.find_all('a', href=re.compile('earnings-call-transcript'))

# get the urls (href elements) from each link
transcript_urls = []
for l in transcript_links:
    if l['href'] not in transcript_urls:   # avoid duplicates
        transcript_urls += [l['href']]

# loop through the links and collect the transcript from each earnings call,
# storing results in a pandas dataframe
############################################################################

# create empty dataframe to collect results
columns = ['link', 'company', 'ticker', 'date', 'quarter', 'transcript']
df = pd.DataFrame({}, columns=columns)


# helper function to get company name, ticker, date, and quarter from the url
def get_url_info(url):
    date = url[url.find('/2023')+1:url.find('/2023')+11]

    url = url[url.find('/2023')+12:]   # other info comes after the date

    # find quarter
    quarter = None
    for q in ['-q1-', '-q2-', '-q3-', '-q4-']:
        if q in url:
            quarter = q[1:3]
            url = url[:url.find(q)]

    # find ticker
    ticker = url[url.rfind('-')+1:]

    # find company
    url = url[:url.rfind('-')]
    company = url

    return [company, ticker, date, quarter]


root = 'https://www.fool.com/'   # root address to append urls to

# loop through urls and get transcript text from each link
for u in transcript_urls:

    if debug:
        print('working on url:', u)

    # create url (root plus url)
    url = root + u

    # get the html
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # get text from transcript body
    body = soup.find(class_='tailwind-article-body')
    transcript_text = body.get_text()

    # create a new row for our dataframe consisting of the url, company name, ticker, date, quarter, and transcript
    # text, and make it a dataframe
    row = [url] + get_url_info(u) + [transcript_text]
    to_append = pd.DataFrame([row], columns=columns)

    # append row to our dataframe
    df = pd.concat([df, to_append])

########################################################################################################################
# PART 2: Scraping interactive webpages (Web 2.0) with Selenium webdriver

# Example: collect pictures of survivor contestants
########################################################################################################################

# import necessary packages
from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
from webdriver_manager.firefox import GeckoDriverManager
from selenium.webdriver.support.ui import Select
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
import time

# read in dataframe with survivor contestant names
data = 'C:/Users/jpkad/Documents/survivor_data/data/'
df = pd.read_csv(data + 'castaways_w_detail.csv')

# get list of castaway names
castaways = list(df['full_name'])

# take first castaway name as a test case
name = castaways[0]
print(name)

# intialize webdriver
ser = Service(r'C:\Program Files\geckodriver-v0.32.2-win32\geckodriver.exe')   # path to geckodriver
options = Options()
options.binary_location = r'C:\Program Files\Mozilla Firefox\firefox.exe'   # path to firefox
driver = webdriver.Firefox(service=ser, options=options)

# go to google
google_url = 'https://www.google.com/'
driver.get(google_url)

# find search input (element tag = <input>, class = "gLFyf")
time.sleep(5)   # wait five seconds to give the website time to load
search_input = driver.find_element(By.CLASS_NAME, 'gLFyf')
time.sleep(1)   # wait another second just to be safe

# input search query (name + "survivor")
search_input.send_keys(name + " survivor")
search_input.send_keys(Keys.ENTER)   # equivalent to hitting 'enter' on keyboard

# we want to get a picture, so let's find the link to get to the image search (use xpath)
# better approach than using time.sleep(): create a WebDriverWait object (this will allow us to wait for items to load
# before clicking or interacting with them)
wait = WebDriverWait(driver, 40)   # wait until condition OR maximum 40 seconds
img_xpath = '/html/body/div[7]/div/div[4]/div/div/div/div[1]/div/a[1]'
img_search = wait.until(EC.visibility_of_element_located((By.XPATH, img_xpath)))
img_search.click()

# get list of image elements (by class name)
images = driver.find_elements(By.CLASS_NAME, 'rg_i')

# save first image of the test case castaway
##################################################
image = images[0]

# get web element that contains the large version of the image when you click on the thumbnail
image.click()
time.sleep(2)
element = driver.find_elements(By.CLASS_NAME, 'v4dQwb')

# get the big version of the image to save
big_img = element[0].find_element(By.CLASS_NAME, 'n3VNCb')

# get url of the big version of the image
image_url = big_img.get_attribute("src")

# write image to file
response = requests.get(image_url, timeout=5)   # set timeout to five seconds if page is not responding
filename = name.replace(" ", "_") + ".jpg"   # name to save image with
with open(filename, "wb") as file:
    file.write(response.content)

# close the webdriver
driver.close()

# full implementation: turn this into a function and loop through castaways, saving first google image of each
###############################################################################################################


def get_img(name, img_directory):
    """
    :param name: name of castaway
    :param img_directory: path of directory where you'd like to save the images
    :return: success (boolean) reporting whether image search and save was successful or not
    """

    # initialize "success" variable as False (this will get updated to True if we make it to the end of the function
    # and successfully save the image file)
    success = False

    search_query = name + ' survivor'

    # standardized search url for images (don't need to click through from google homepage every time)
    search_url = f"https://www.google.com/search?site=&tbm=isch&source=hp&biw=1873&bih=990&q={search_query}"

    # get image search page
    driver.get(search_url)
    time.sleep(5)

    # get first image
    try:
        image = driver.find_element(By.CLASS_NAME, 'rg_i')
    except NoSuchElementException:
        print('ERROR: not able to find an image for', name)
        return success

    # get web element that contains the large version of the image when you click on the thumbnail
    image.click()
    time.sleep(2)
    try:
        element = driver.find_elements(By.CLASS_NAME, 'v4dQwb')
    except NoSuchElementException:
        print('ERROR: not able to find element containing enlarged image')
        return success

    # get the big version of the image to save
    try:
        big_img = element[0].find_element(By.CLASS_NAME, 'n3VNCb')
    except NoSuchElementException:
        print('ERROR: not able to find enlarged version of image')
        return success

    # get url of the big version of the image
    image_url = big_img.get_attribute("src")

    # write image to file
    try:
        response = requests.get(image_url, timeout=5)  # set timeout to five seconds if page is not responding

    # sometimes large image link lives in second element with class name 'v4dQwb'... Try this if first image url isn't
    # working
    except requests.exceptions.InvalidSchema:
        big_img = element[1].find_element(By.CLASS_NAME, 'n3VNCb')
        image_url = big_img.get_attribute("src")
        try:
            response = requests.get(image_url, timeout=5)
        except:
            print('ERROR: problem with image url for', name)
            return success

    # write image to file
    filename = img_directory + '/' + name.replace(" ", "_") + ".jpg"  # name to save image with
    with open(filename, "wb") as file:
        file.write(response.content)

    success = True

    return success


# initialize webdriver
driver = webdriver.Firefox(service=ser, options=options)

# specify directory where you want to save the images
img_folder = r'C:\Users\jpkad\OneDrive\Pictures'

# create a new variable in the dataframe to capture whether image was saved or not
df['image_saved'] = 'no'

# collect images for castaways 0 to 5 and record in dataframe whether we were able to save an image for them
for c in castaways[0:5]:
    if debug:
        print('working on castaway:', c)

    # function returns whether image search was successful
    success = get_img(c, img_folder)
    if success:
        # update dataframe to record whether we were able to save an image for this castaway or not
        df.loc[df['full_name'] == c, 'image_saved'] = 'yes'

# close driver
driver.close()




