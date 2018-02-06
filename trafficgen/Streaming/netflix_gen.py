from selenium import webdriver
import time
from random import randint
from trafficgen.Streaming.capture import *
import _thread
import datetime

def login(username, password):
    browser.get("http://www.netflix.com/login")
    # LOGIN
    emailField = browser.find_element_by_css_selector("#email")
    passwordField = browser.find_element_by_css_selector("#password")
    emailField.send_keys(username)
    passwordField.send_keys(password)
    submit = browser.find_element_by_css_selector("#appMountPoint > div > div.login-body > div > div > form:nth-child(2) > button")
    submit.click()
    # Change to correct profile
    browser.get("https://www.netflix.com/SwitchProfile?tkn=I42P4G75VVDM7LV626VKTXTXGI")

def streamVideo(duration):
        # Choose random video
    videos = browser.find_elements_by_css_selector("div.title-card-container a[href*='/watch/']")

    video = videos[randint(0,len(videos))]

    link = video.get_attribute('href')
    browser.get(link)
    time.sleep(duration)




username = "alex_mulan@hotmail.com"
password = "Basser77"



for i in range(5):

    # Open the chrome webdriver
    path = "C:\\Users\\arhjo\\OneDrive\\DTU\\Thesis\\classification-of-encrypted-traffic\\trafficgen\\Streaming\\chromedriver.exe"
    options = webdriver.ChromeOptions()
    #options.set_headless()
    browser = webdriver.Chrome(executable_path=path, chrome_options=options)
    browser.minimize_window()
    now = datetime.datetime.now()

    #Specify duration in seconds
    duration = 60 * 10

    print("Started capture at: %s:%s" % (now.hour,now.minute))
    print("Duration is %d" % (duration/60))
    _thread.start_new_thread(captureTraffic,(1, duration, 'C:/users/arhjo/desktop/test', "netflix"))


    login(username, password)
    streamVideo(duration)


    # Close browser
    browser.close()
    time.sleep(30)


