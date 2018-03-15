import time
from random import randint

class Streaming:

    def streamVideo(self,browser):
        pass


class twitch(Streaming):

    def streamVideo(self, browser):
        browser.get('https://www.twitch.tv/directory/game/League%20of%20Legends/videos/all')
        time.sleep(2)
        # Choose random video
        videos = browser.find_elements_by_css_selector("a[href*='/videos/']")

        video = videos[randint(0,len(videos))]

        link = video.get_attribute('href')
        browser.get(link)


class netflix(Streaming):

    def streamVideo(self, browser):
        # Change to correct profile
        browser.get("https://www.netflix.com/SwitchProfile?tkn=I42P4G75VVDM7LV626VKTXTXGI")
        # Choose random video
        videos = browser.find_elements_by_css_selector("div.title-card-container a[href*='/watch/']")

        video = videos[randint(0, len(videos))]

        link = video.get_attribute('href')
        browser.get(link)


class hbo(Streaming):

    def streamVideo(self,browser):
        browser.get("https://dk.hbonordic.com/home")
        time.sleep(2)
        videos = browser.find_elements_by_css_selector("a[data-automation='play-button']")
        video = videos[randint(0, len(videos))]
        videoURL = video.get_attribute("href")
        browser.get(videoURL)

class youtube(Streaming):

    def streamVideo(self,browser):
        browser.get('https://www.youtube.com')
        # Choose random video
        videos = browser.find_elements_by_css_selector("ytd-grid-video-renderer  a[href*='/watch?v=']")

        video = videos[randint(0, len(videos))]

        link = video.get_attribute('href')

        browser.get(link)


class dr(Streaming):

    def streamVideo(self,browser):
        browser.get('https://www.dr.dk/tv')
        # Choose random video
        videos = browser.find_elements_by_class_name('program-link')

        video = videos[randint(0, len(videos))]

        link = video.get_attribute('href')
        browser.get(link)

        time.sleep(2)
        play_button = browser.find_element_by_css_selector('button[title="Afspil"]')
        play_button.click()