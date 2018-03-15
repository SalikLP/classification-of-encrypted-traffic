import datetime
from threading import Thread
from selenium import webdriver
from slackclient import SlackClient
import traceback
from selenium.webdriver.support.ui import WebDriverWait

import trafficgen.Streaming.win_capture as cap
import trafficgen.Streaming.streaming_types as stream

# import unix_capture as cap
# import drtv_gen as dr
# import hbonordic_gen as hbo
# import netflix_gen as netflix
# import youtube_gen as youtube
# import twitch_gen as twitch


def notifySlack(message):
    sc = SlackClient(slack_token)
    try:
        sc.api_call("chat.postMessage", channel="#server", text=message)
    except:
        sc.api_call("chat.postMessage", channel="#server", text="Could not send stacktrace")


def generate_streaming(duration, dir, total_iterations, chrome_options=None):
    iterations = 0
    while iterations < total_iterations:
        '''
        try:
            if iterations % 2 == 0:
                browsers, capture_thread, file, streaming_threads = generate_multithreaded_dr_streaming(dir, duration)
            else:
                browsers, capture_thread, file, streaming_threads = generate_multithreaded_youtube_streaming(dir, duration)
        except Exception as ex:
            notifySlack("Something went wrong when setting up the threads \n %s" % traceback.format_exc())
        '''
        browsers, capture_thread, file, streaming_threads = generate_singlethreaded_streaming(stream.HboNordic, "hbo", dir, duration, chrome_options)
        # browsers, capture_thread, file, streaming_threads = generate_multithreaded_streaming(stream.Twitch, "twitch", dir, duration, chrome_options)
        try:
            capture_thread.start()
            for thread in streaming_threads:
                # Start streaming threads
                thread.start()
            print("streaming started")
            capture_thread.join() # Stream until the capture thread joins
            print("capture done - thread has joined")
            for browser in browsers:
                clear_cache(browser)
                browser.close()

        except Exception as e:
            notifySlack("Something went wrong %s" % traceback.format_exc())
            # Wait for capture thread
            capture_thread.join()
            # Do a cleanup since somthing went wrong
            cap.cleanup(file)
            for browser in browsers:
                browser.close()
        for browser in browsers:
            browser.quit()
        iterations += 1


def generate_singlethreaded_streaming(obj: stream.Streaming, stream_name, dir, duration, chrome_options=None):
    #### STREAMING ####
    # Create filename
    now = datetime.datetime.now()
    file = dir + "/%s-%.2d%.2d_%.2d%.2d%.2d.pcap" % (stream_name, now.day, now.month, now.hour, now.minute, now.second)
    # Instantiate thread
    capture_thread = Thread(target=cap.captureTraffic, args=(5, duration, dir, file))
    # Create single thread for streaming
    streaming_threads = []
    browsers = []
    browser = webdriver.Chrome(options=chrome_options)
    browser.implicitly_wait(10)
    browsers.append(browser)
    t1 = Thread(target=obj.stream_video, args=(obj, browser))
    streaming_threads.append(t1)
    return browsers, capture_thread, file, streaming_threads


def generate_multithreaded_streaming(obj: stream.Streaming, stream_name, dir, duration, chrome_options=None):
    #### STREAMING ####
    # Create filename
    now = datetime.datetime.now()
    file = dir + "/%s-%.2d%.2d_%.2d%.2d%.2d.pcap" % (stream_name, now.day, now.month, now.hour, now.minute, now.second)
    # Instantiate thread
    capture_thread = Thread(target=cap.captureTraffic, args=(5, duration, dir, file))
    # Create five threads for streaming
    streaming_threads = []
    browsers = []
    for i in range(5):
        browser = webdriver.Chrome(options=chrome_options)
        browser.implicitly_wait(10)
        browsers.append(browser)
        t = Thread(target=obj.stream_video, args=(obj, browser))
        streaming_threads.append(t)

    return browsers, capture_thread, file, streaming_threads


def get_clear_browsing_button(driver):
    """Find the "CLEAR BROWSING BUTTON" on the Chrome settings page."""
    return driver.find_element_by_css_selector('* /deep/ #clearBrowsingDataConfirm')


def clear_cache(driver, timeout=60):
    """Clear the cookies and cache for the ChromeDriver instance."""
    # navigate to the settings page
    driver.get('chrome://settings/clearBrowserData')

    # wait for the button to appear
    wait = WebDriverWait(driver, timeout)
    wait.until(get_clear_browsing_button)

    # click the button to clear the cache
    get_clear_browsing_button(driver).click()

    # wait for the button to be gone before returning
    wait.until_not(get_clear_browsing_button)


if __name__ == "__main__":
    #netflixuser = os.environ["netflixuser"]
    #netflixpassword = os.environ["netflixpassword"]
    #hbouser = os.environ["hbouser"]
    #hbopassword = os.environ["hbopassword"]
    #slack_token = os.environ['slack_token']
    # Specify duration in seconds
    duration = 30 * 1
    total_iterations = 1000
    save_dir = 'D:\\Data'
    chrome_profile_dir = 'C:\\Users\\salik\\AppData\\Local\\Google\\Chrome\\User Data\\Profile 3'
    options = webdriver.ChromeOptions()
    options.add_argument('user-data-dir=' + chrome_profile_dir)

    generate_streaming(duration, save_dir, total_iterations, options)
    print("something")