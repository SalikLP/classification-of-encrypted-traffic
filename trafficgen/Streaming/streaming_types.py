from random import randint


class Streaming:

    def stream_video(self, browser):
        pass


class Twitch(Streaming):

    def stream_video(self, browser):
        browser.get('https://www.twitch.tv/directory/game/League%20of%20Legends/videos/all')
        # Choose random video
        videos = browser.find_elements_by_css_selector("a[href*='/videos/']")
        video = videos[randint(0, len(videos))]
        link = video.get_attribute('href')
        browser.get(link)


class Youtube(Streaming):

    def stream_video(self, browser):
        browser.get('https://www.youtube.com')
        # Choose random video
        videos = browser.find_elements_by_css_selector("ytd-grid-video-renderer  a[href*='/watch?v=']")
        video = videos[randint(0, len(videos))]
        # thumbnail
        link = video.get_attribute('href')
        browser.get(link)


class Netflix(Streaming):

    def stream_video(self, browser):
        # Change to correct profile
        browser.get("https://www.netflix.com/SwitchProfile?tkn=I42P4G75VVDM7LV626VKTXTXGI")
        # Choose random video
        videos = browser.find_elements_by_css_selector("div.title-card-container a[href*='/watch/']")
        video = videos[randint(0, len(videos))]
        link = video.get_attribute('href')
        browser.get(link)


class DrTv(Streaming):

    def stream_video(self, browser):
        browser.get('https://www.dr.dk/tv')
        # Choose random video
        videos = browser.find_elements_by_class_name('program-link')
        video = videos[randint(0, len(videos))]
        link = video.get_attribute('href')
        browser.get(link)
        play_button = browser.find_element_by_css_selector('button[title="Afspil"]')
        play_button.click()


class HboNordic(Streaming):
    def stream_video(self, browser):
        browser.get("https://dk.hbonordic.com/home")
        videos = browser.find_elements_by_css_selector("a[data-automation='play-button']")
        video = videos[randint(0, len(videos))]
        video_url = video.get_attribute("href")
        browser.get(video_url)
