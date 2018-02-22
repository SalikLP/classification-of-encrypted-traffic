import time
from random import randint


def streamVideo(browser, duration):
    browser.get('https://www.youtube.com')
    # Choose random video
    videos = browser.find_elements_by_css_selector("ytd-grid-video-renderer  a[href*='/watch?v=']")

    video = videos[randint(0,len(videos))]

    # thumbnail

    link = video.get_attribute('href')

    browser.get(link)