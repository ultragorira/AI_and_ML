import logging
import agentql
from playwright.sync_api import sync_playwright
from transformers import pipeline
from collections import Counter

sentiment_pipeline = pipeline("sentiment-analysis")
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)
URL = "https://www.youtube.com"

with sync_playwright() as playwright, playwright.chromium.launch(headless=False) as browser:
    page = agentql.wrap(browser.new_page())
    page.goto(URL)

    QUERY = """
        {
    cookies_form {
        reject_btn
                }
        }
        """

    SEARCH_QUERY = """
    {
        search_input
        search_btn
    }
    """

    VIDEO_QUERY = """
    {
        videos[] {
            video_link
            video_title
            channel_name
        }
    }
    """

    VIDEO_CONTROL_QUERY = """
    {
        play_or_pause_btn
        expand_description_btn
    }
    """

    DESCRIPTION_QUERY = """
    {
        description_text
    }
    """

    COMMENT_QUERY = """
    {
        comments[] {
            channel_name
            comment_text
        }
    }
    """

    try:

        response = page.query_elements(QUERY)

        if response.cookies_form.reject_btn != None:
            response.cookies_form.reject_btn.click()

        response = page.query_elements(SEARCH_QUERY)
        response.search_input.type("Lego Barad Dur Review", delay=75)
        response.search_btn.click()

        response = page.query_elements(VIDEO_QUERY)
        log.debug(f"Clicking Youtube Video: {response.videos[0].video_title.text_content()}")
        response.videos[0].video_link.click()  # click the first youtube video

        response = page.query_elements(VIDEO_CONTROL_QUERY)
        response.expand_description_btn.click()

        response_data = page.query_data(DESCRIPTION_QUERY)
        log.debug(f"Captured the following description: \n{response_data['description_text']}")

        for _ in range(7):
            page.keyboard.press("PageDown")
            page.wait_for_page_ready_state()

        response = page.query_data(COMMENT_QUERY)
        log.debug(f"Captured {len(response.get("comments"))} comments!")

        print(response["comments"])

        print("SENTIMENT ANALYSIS:")
        sentiment = sentiment_pipeline([comment["comment_text"] for comment in response["comments"]])
        labels_count = Counter([d['label'] for d in sentiment])
        most_common_label = labels_count.most_common(1)[0][0]

        print(most_common_label)

    except Exception as e:
        log.error(f"Found Error: {e}")
        raise e
