import datetime
from threading import Thread
from selenium import webdriver
from slackclient import SlackClient
import traceback
import os
'''
import trafficgen.Streaming.capture as cap
import trafficgen.Streaming.drtv_gen as dr
import trafficgen.Streaming.hbonordic_gen as hbo
import trafficgen.Streaming.netflix_gen as netflix
import trafficgen.Streaming.youtube_gen as youtube
'''
import capture as cap
import drtv_gen as dr
import hbonordic_gen as hbo
import netflix_gen as netflix
import youtube_gen as youtube

def notifySlack(message):
    sc = SlackClient(slack_token)
    try:
        sc.api_call("chat.postMessage", channel="#server", text=message)
    except:
        sc.api_call("chat.postMessage", channel="#server", text="Could not send stacktrace")

def generate_streaming(duration):
    # Open the chrome webdriver
    dir = '/home/mclrn/Data'

    dr_counter = 0
    while dr_counter < 100:
        browser = webdriver.Chrome()
        # Create filename
        now = datetime.datetime.now()
        file = dir + "/%s-%.2d%.2d_%.2d%.2d.pcap" % ("drtv", now.day, now.month, now.hour, now.minute)
        # Instantiate thread
        t1 = Thread(target=cap.captureTraffic, args=(1, duration, dir, file))
        try:
            t1.start()
            dr.streamVideo(browser, duration)
            browser.close()
            dr_counter += 1
        except Exception as e:
            t1.join() # Wait for thread to join
            cap.cleanup(file) # Cleanup by deleting the pcap file
            browser.close() # Close the browser
            notifySlack("Something went wrong while streaming \n %s" % traceback.format_exc())
        browser.quit()

    netflix_counter = 0
    while netflix_counter < 100:
        browser = webdriver.Chrome()
        # Create filename
        now = datetime.datetime.now()
        file = dir + "/%s-%.2d%.2d_%.2d%.2d.pcap" % ("netflix", now.day, now.month, now.hour, now.minute)
        # Instantiate thread
        t1 = Thread(target=cap.captureTraffic, args=(1, duration, dir, file))
        try:
            t1.start()
            netflix.streamVideo(browser, duration, netflixuser, netflixpassword)
            browser.close()
            netflix_counter += 1
        except Exception as e:
            t1.join()  # Wait for thread to join
            cap.cleanup(file)  # Cleanup by deleting the pcap file
            browser.close()  # Close the browser
            notifySlack("Something went wrong while streaming \n %s" % traceback.format_exc())
        browser.quit()

    youtube_counter = 0
    while youtube_counter < 100:
        browser = webdriver.Chrome()
        # Create filename
        now = datetime.datetime.now()
        file = dir + "/%s-%.2d%.2d_%.2d%.2d.pcap" % ("youtube", now.day, now.month, now.hour, now.minute)
        # Instantiate thread
        t1 = Thread(target=cap.captureTraffic, args=(1, duration, dir, file))
        try:
            t1.start()
            youtube.streamVideo(browser, duration)
            browser.close()
            youtube_counter += 1
        except Exception as e:
            t1.join()  # Wait for thread to join
            cap.cleanup(file)  # Cleanup by deleting the pcap file
            browser.close()  # Close the browser
            notifySlack("Something went wrong while streaming \n %s" % traceback.format_exc())
        browser.quit()

    hbo_counter = 0
    while hbo_counter < 100:
        browser = webdriver.Chrome()
        # Create filename
        now = datetime.datetime.now()
        file = dir + "/%s-%.2d%.2d_%.2d%.2d.pcap" % ("hbo", now.day, now.month, now.hour, now.minute)
        # Instantiate thread
        t1 = Thread(target=cap.captureTraffic, args=(1, duration, dir, file))
        try:
            t1.start()
            hbo.streamVideo(browser, duration, hbouser, hbopassword)
            browser.close()
            hbo_counter += 1
        except Exception as e:
            t1.join()  # Wait for thread to join
            cap.cleanup(file)  # Cleanup by deleting the pcap file
            browser.close()  # Close the browser
            notifySlack("Something went wrong while streaming \n %s" % traceback.format_exc())
        browser.quit()

    notifySlack("Streaming has completed")


def generate_streaming_multithread(duration):
    total_iterations = 100
    iterations = 0
    dir = '/home/mclrn/Data'
    while iterations < total_iterations:

        #### DRTV STREAMING ####
        # Create filename
        now = datetime.datetime.now()
        file = dir + "/%s-%.2d%.2d_%.2d%.2d.pcap" % ("drtv", now.day, now.month, now.hour, now.minute)
        # Instantiate thread
        capture_dr_thread = Thread(target=cap.captureTraffic, args=(1, duration, dir, file))

        # Create five threads for streaming DR
        streaming_threads = []
        browsers = []

        browser1 = webdriver.Chrome()
        browsers.append(browser1)
        dr1 = Thread(target=dr.streamVideo, args=(browser1,duration))
        streaming_threads.append(dr1)

        browser2 = webdriver.Chrome()
        browsers.append(browser2)
        dr2 = Thread(target=dr.streamVideo, args=(browser2, duration))
        streaming_threads.append(dr2)

        browser3 = webdriver.Chrome()
        browsers.append(browser3)
        dr3 = Thread(target=dr.streamVideo, args=(browser3, duration))
        streaming_threads.append(dr3)

        browser4 = webdriver.Chrome()
        browsers.append(browser4)
        dr4 = Thread(target=dr.streamVideo, args=(browser4, duration))
        streaming_threads.append(dr4)

        browser5 = webdriver.Chrome()
        browsers.append(browser5)
        dr5 = Thread(target=dr.streamVideo, args=(browser5, duration))
        streaming_threads.append(dr5)

        try:
            capture_dr_thread.start()
            for thread in streaming_threads:
                # Start streaming threads
                thread.start()

            for thread in streaming_threads:
                # Wait until all threads have joined
                thread.join()

            for browser in browsers:
                browser.close()

        except Exception as e:
            for thread in streaming_threads:
                # Wait for all threads
                thread.join()
            # Wait for capture thread
            capture_dr_thread.join()
            cap.cleanup(file)
            for browser in browsers:
                browser.close()
            notifySlack("Something went wrong %s" % traceback.format_exc())
        for browser in browsers:
            browser.quit()



if __name__ == "__main__":

    '''
    netflixuser = os.environ["netflixuser"]
    netflixpassword = os.environ["netflixpassword"]

    hbouser = os.environ["hbouser"]
    hbopassword = os.environ["hbopassword"]
    '''
    # Specify duration in seconds
    duration = 60 * 1
    #generate_streaming(duration)
    generate_streaming_multithread(duration)