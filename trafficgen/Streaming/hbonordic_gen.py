import time
from random import randint

def expand_shadow_element(browser, element):
  shadow_root = browser.execute_script('return arguments[0].shadowRoot', element)
  return shadow_root

def enable_flash(browser, page):
    browser.get("chrome://settings/content/flash")
    time.sleep(2)
    settings = browser.find_elements_by_tag_name("settings-ui")[0]
    shadow_settings = expand_shadow_element(browser, settings)
    settings_main = shadow_settings.find_elements_by_css_selector("settings-main")[0]
    shadow_settings_main = expand_shadow_element(browser, settings_main)
    settings_basic_page = shadow_settings_main.find_elements_by_css_selector("settings-basic-page")[0]
    shadow_settings_basic_page = expand_shadow_element(browser, settings_basic_page)
    advancedPage = shadow_settings_basic_page.find_elements_by_css_selector("div[id='advancedPage']")[0]
    settingsSection = advancedPage.find_elements_by_css_selector("settings-section[section='privacy']")[0]
    settings_privacy_page = settingsSection.find_elements_by_css_selector("settings-privacy-page")[0]
    shadow_settings_privacy_page = expand_shadow_element(browser, settings_privacy_page)
    settings_subpage = shadow_settings_privacy_page.find_elements_by_css_selector("settings-subpage")[0]
    category_setting_exceptions = settings_subpage.find_elements_by_css_selector("category-setting-exceptions")[0]
    shadow_category_setting_exceptions = expand_shadow_element(browser, category_setting_exceptions)
    site_list = shadow_category_setting_exceptions.find_elements_by_css_selector("site-list[category-header='Allow']")[
        0]
    shadow_site_list = expand_shadow_element(browser, site_list)
    addSite = shadow_site_list.find_element_by_id('addSite')
    addSite.click()
    dialog = shadow_site_list.find_elements_by_css_selector("add-site-dialog")[0]
    shadow_dialog = expand_shadow_element(browser, dialog)
    dialog_window = shadow_dialog.find_element_by_id("dialog")
    input_box = dialog_window.find_element_by_id("site")
    print("Enabling flash for page: %s" %page)
    input_box.send_keys("dk.hbonordic.com:443")  # INSERT EXCEPTION HERE
    time.sleep(2)
    add_button = dialog_window.find_element_by_id("add")
    add_button.click()
    id = browser.session_id
    browser.close()
    time.sleep(2)
    browser.launch_app(id)


def login(browser, username, password):
    browser.get("https://dk.hbonordic.com/sign-in")
    # LOGIN
    time.sleep(3)
    emailField = browser.find_element_by_id("email")
    passwordField = browser.find_element_by_css_selector("input[data-automation='sign-in-password-input-input']")
    emailField.send_keys(username)
    passwordField.send_keys(password)

    submit = browser.find_element_by_css_selector("button[data-automation='sign-in-submit-button']")

    submit.click()

def streamVideo(browser):
    browser.get("https://dk.hbonordic.com/home")
    time.sleep(2)
    videos = browser.find_elements_by_css_selector("a[data-automation='play-button']")
    video = videos[randint(0, len(videos))]
    videoURL = video.get_attribute("href")
    browser.get(videoURL)
