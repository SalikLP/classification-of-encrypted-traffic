import time
from random import randint


def streamVideo(browser, duration):
    browser.get('https://www.dr.dk/tv')
    # Choose random video
    videos = browser.find_elements_by_class_name('program-link')


    video = videos[randint(0,len(videos))]

    link = video.get_attribute('href')
    browser.get(link)

    time.sleep(2)
    play_button = browser.find_element_by_css_selector('button[title="Afspil"]')
    play_button.click()


