from slackclient import SlackClient
import os

slack_token = os.environ["SLACK_TOKEN"]
sc = SlackClient(slack_token)


def send_message(text):
    sc.api_call(
      "chat.postMessage",
      channel="#mclrn",
      text=str(text)
    )
