import datetime
from threading import Thread
from selenium import webdriver
from slackclient import SlackClient
import traceback

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
        browsers, capture_thread, file, streaming_threads = generate_multithreaded_streaming(stream.Twitch, "twitch", dir, duration, chrome_options)
        try:
            capture_thread.start()
            for thread in streaming_threads:
                # Start streaming threads
                thread.start()
            print("streaming started")
            capture_thread.join() # Stream until the capture thread joins
            print("capture done - thread has joined")
            for browser in browsers:
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
    file = dir + "/%s-%.2d%.2d_%.2d%.2d.pcap" % (stream_name, now.day, now.month, now.hour, now.minute)
    # Instantiate thread
    capture_thread = Thread(target=cap.captureTraffic, args=(1, duration, dir, file))
    # Create single thread for streaming
    streaming_threads = []
    browsers = []
    browser1 = webdriver.Chrome(options=chrome_options)
    browsers.append(browser1)
    t1 = Thread(target=obj.stream_video, args=(obj, browser1))
    streaming_threads.append(t1)
    return browsers, capture_thread, file, streaming_threads


def generate_multithreaded_streaming(obj: stream.Streaming, stream_name, dir, duration, chrome_options=None):
    #### STREAMING ####
    # Create filename
    now = datetime.datetime.now()
    file = dir + "/%s-%.2d%.2d_%.2d%.2d.pcap" % (stream_name, now.day, now.month, now.hour, now.minute)
    # Instantiate thread
    capture_thread = Thread(target=cap.captureTraffic, args=(1, duration, dir, file))
    # Create five threads for streaming
    streaming_threads = []
    browsers = []
    browser1 = webdriver.Chrome(options=chrome_options)
    browsers.append(browser1)
    t1 = Thread(target=obj.stream_video, args=(obj, browser1))
    streaming_threads.append(t1)
    browser2 = webdriver.Chrome(options=chrome_options)
    browsers.append(browser2)
    t2 = Thread(target=obj.stream_video, args=(obj, browser2))
    streaming_threads.append(t2)
    browser3 = webdriver.Chrome(options=chrome_options)
    browsers.append(browser3)
    t3 = Thread(target=obj.stream_video, args=(obj, browser3))
    streaming_threads.append(t3)
    browser4 = webdriver.Chrome(options=chrome_options)
    browsers.append(browser4)
    t4 = Thread(target=obj.stream_video, args=(obj, browser4))
    streaming_threads.append(t4)
    browser5 = webdriver.Chrome(options=chrome_options)
    browsers.append(browser5)
    t5 = Thread(target=obj.stream_video, args=(obj, browser5))
    streaming_threads.append(t5)
    return browsers, capture_thread, file, streaming_threads


if __name__ == "__main__":
    #netflixuser = os.environ["netflixuser"]
    #netflixpassword = os.environ["netflixpassword"]
    #hbouser = os.environ["hbouser"]
    #hbopassword = os.environ["hbopassword"]
    #slack_token = os.environ['slack_token']
    # Specify duration in seconds
    duration = 30 * 1
    total_iterations = 100
    save_dir = '/home/mclrn/Data'
    chrome_profile_dir = "/home/mclrn/.config/google-chrome/"
    options = webdriver.ChromeOptions()
    options.add_argument('user-data-dir=' + chrome_profile_dir)

    generate_streaming(duration, save_dir, total_iterations, options)
    print("something")