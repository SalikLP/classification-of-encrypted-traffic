import time
from random import randint

def login(browser, username, password):


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

def streamVideo(browser, duration, username, password):
    login(browser, username, password)
    # Choose random video
    videos = browser.find_elements_by_css_selector("div.title-card-container a[href*='/watch/']")

    video = videos[randint(0,len(videos))]

    link = video.get_attribute('href')
    browser.get(link)
    time.sleep(duration)






