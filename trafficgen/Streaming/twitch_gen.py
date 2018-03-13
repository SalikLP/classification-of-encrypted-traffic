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


