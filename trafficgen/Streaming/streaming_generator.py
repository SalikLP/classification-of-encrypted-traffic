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
    slack_token = "xoxp-293516421953-294177990549-318003849223-df8cf2c41f0167d74cc779a09edf450b"
    sc = SlackClient(slack_token)
    try:
        sc.api_call("chat.postMessage", channel="#server", text=message)
    except:
        sc.api_call("chat.postMessage", channel="#server", text="Could not send stacktrace")

def generate_streaming(duration):
    # Open the chrome webdriver
    dir = '/home/mclrn/Data'
    #TODO: These could be done with while loops (while dr < 100: streamDR) then we could sorround each with a try/catch and thereby only increment the counter if the streaming is successful
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
    notifySlack("Streaming has completed")
'''
    try:
        for i in range(10): # This forloop is to make easy printing of the progress
            for j in range(5):
                browser = webdriver.Chrome(options=options)
                _thread.start_new_thread(captureTraffic, (1, duration, '/home/mclrn/Data', "drtv"))
                dr.streamVideo(browser, duration)
                browser.close()

                browser = webdriver.Chrome(options=options)
                _thread.start_new_thread(captureTraffic, (1, duration, '/home/mclrn/Data', "netflix"))
                netflix.streamVideo(browser, duration, netflixuser, netflixpassword)
                browser.close()
                # TODO: The hbo streaming can only be generated if we run in non-headless mode on the server. Find out how we can do that, it currently throws an error.
                
                
                browser = webdriver.Chrome(options=options)
                _thread.start_new_thread(captureTraffic, (1, duration, '/home/mclrn/Data', "hbo"))
                hbo.streamVideo(browser, duration,hbouser,hbopassword)
                browser.close()
            
                
                browser = webdriver.Chrome(options=options)
                _thread.start_new_thread(captureTraffic, (1, duration, '/home/mclrn/Data', "youtube"))
                youtube.streamVideo(browser, duration)
                browser.close()
            notifySlack("Streaming generation is %d0%% done" % (i+1))
    except Exception as e:
        #TODO: Maybe we can do some cleanup, remove the pcap file that was created. We might also try to restart the process in some way.
        notifySlack("Something went wrong while streaming \n %s" % traceback.format_exc())
    

'''












if __name__ == "__main__":
    netflixuser = os.environ["netflixuser"]
    netflixpassword = os.environ["netflixpassword"]


    hbouser = os.environ["hbouser"]
    hbopassword = os.environ["hbopassword"]

    # Specify duration in seconds
    duration = 60 * 2
    generate_streaming(duration)