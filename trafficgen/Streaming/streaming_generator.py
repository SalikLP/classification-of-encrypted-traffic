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
    '''
    This method is not used anymore

    :param duration: time in seconds for streaming
    :return:
    '''
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



def generate_streaming_netflix(duration, username, password):
    '''
    This method will generate total_iterations of pcap files containing traffic captured while streaming netflix.
    :param duration: Time to stream
    :param username: Valid netflix username
    :param password: Password for netflix account
    :return:
    '''
    total_iterations = 1000
    iteration = 0
    dir = '/home/mclrn/Data'


    # instantiate browser
    browser = webdriver.Chrome()
    # Perform login procedure in the browser
    netflix.login(browser, username, password)

    while iteration < total_iterations:
        # Create filename
        now = datetime.datetime.now()
        file = dir + "/%s-%.2d%.2d_%.2d%.2d.pcap" % ("netflix", now.day, now.month, now.hour, now.minute)
        # Instantiate thread for capturing traffic
        t1 = Thread(target=cap.captureTraffic, args=(1, duration, dir, file))

        try:
            t1.start()
            # Start stream
            netflix.streamVideo(browser, duration)
            # Wait until the capture is done
            t1.join()
            iteration +=1
        except Exception as e:
            notifySlack("Something went wrong in the attempt to stream netflix \n %s" % traceback.format_exc())
            t1.join()
            cap.cleanup(file)

    notifySlack("The server has ended streaming from netflix. Iterations completed %d" %iteration)

def generate_streaming_multithread(duration):
    total_iterations = 100
    iterations = 0
    dir = '/home/mclrn/Data'

    streaming_threads = []
    browsers = []
    file = " "
    while iterations < total_iterations:

        try:
            if iterations % 2 == 0:
                browsers, capture_thread, file, streaming_threads = generate_multithreaded_dr_streaming(dir, duration)
            else:
                browsers, capture_thread, file, streaming_threads = generate_multithreaded_youtube_streaming(dir, duration)
        except Exception as ex:
            notifySlack("Something went wrong when setting up the threads \n %s" % traceback.format_exc())

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
        iterations+=1

def generate_multithreaded_dr_streaming(dir, duration):
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
    dr1 = Thread(target=dr.streamVideo, args=(browser1, duration))
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
    return browsers, capture_dr_thread, file, streaming_threads


def generate_multithreaded_youtube_streaming(dir, duration):
    #### YOUTUBE STREAMING ####
    # Create filename
    now = datetime.datetime.now()
    file = dir + "/%s-%.2d%.2d_%.2d%.2d.pcap" % ("youtube", now.day, now.month, now.hour, now.minute)
    # Instantiate thread
    capture_youtube_thread = Thread(target=cap.captureTraffic, args=(1, duration, dir, file))
    # Create five threads for streaming youtube
    streaming_threads = []
    browsers = []
    browser1 = webdriver.Chrome()
    browsers.append(browser1)
    youtube1 = Thread(target=youtube.streamVideo, args=(browser1, duration))
    streaming_threads.append(youtube1)
    browser2 = webdriver.Chrome()
    browsers.append(browser2)
    youtube2 = Thread(target=youtube.streamVideo, args=(browser2, duration))
    streaming_threads.append(youtube2)
    browser3 = webdriver.Chrome()
    browsers.append(browser3)
    youtube3 = Thread(target=youtube.streamVideo, args=(browser3, duration))
    streaming_threads.append(youtube3)
    browser4 = webdriver.Chrome()
    browsers.append(browser4)
    youtube4 = Thread(target=youtube.streamVideo, args=(browser4, duration))
    streaming_threads.append(youtube4)
    browser5 = webdriver.Chrome()
    browsers.append(browser5)
    youtube5 = Thread(target=youtube.streamVideo, args=(browser5, duration))
    streaming_threads.append(youtube5)
    return browsers, capture_youtube_thread, file, streaming_threads






if __name__ == "__main__":
    #netflixuser = os.environ["netflixuser"]
    #netflixpassword = os.environ["netflixpassword"]
    #hbouser = os.environ["hbouser"]
    #hbopassword = os.environ["hbopassword"]
    #slack_token = os.environ['slack_token']
    # Specify duration in seconds
    duration = 60 * 1
    #generate_streaming(duration)
    #generate_streaming_multithread(duration)
    generate_streaming_netflix(duration,netflixuser, netflixpassword)