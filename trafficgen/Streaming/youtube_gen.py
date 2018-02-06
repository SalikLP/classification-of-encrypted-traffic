from selenium import webdriver
import time
from random import randint
from trafficgen.Streaming.capture import *
import _thread


def streamVideo(duration):
    try:
        browser.get('https://www.youtube.com')
        # Choose random video
        videos = browser.find_elements_by_css_selector("ytd-grid-video-renderer  a[href*='/watch?v=']")

        video = videos[randint(0,len(videos))]

        # thumbnail

        link = video.get_attribute('href')


        videoLengthElem = video.find_element_by_css_selector('span[aria-label*="minutter"]')
        videoLength = videoLengthElem.text.split(':')[0]
        print("Video length: %s" % videoLength)
        if(videoLength > duration/60): # We should only stream the video if it is long enough
            browser.get(link)
            time.sleep(duration)
    except:
        print("Something went wrong")
#overlays > ytd-thumbnail-overlay-time-status-renderer > span
for i in range(5):
    # Open the chrome webdriver
    browser = webdriver.Chrome("C:\\Users\\arhjo\\OneDrive\\DTU\\Thesis\\classification-of-encrypted-traffic\\trafficgen\\Streaming\\chromedriver.exe")
    browser.minimize_window()
    #Specify duration in seconds
    duration = 60 * 5
    #_thread.start_new_thread(captureTraffic,(1, duration, 'C:/users/arhjo/desktop/test', "youtube"))

    streamVideo(duration)


    # Close browser
    browser.close()

