import time


def hide_sdk_banner(driver, sleep=2.0, include_placeholder=True):
    time.sleep(sleep)
    driver.execute_script("document.getElementById('onetrust-banner-sdk').style.display='none';")
    if include_placeholder:
        driver.execute_script("document.getElementsByClassName('otPlaceholder')[0].style.display='none';")


def hide_advert_banner(driver, sleep=2):
    time.sleep(sleep)
    driver.execute_script(
        "document.getElementsByClassName('boxOverContent boxOverContent--type-2 isSticky isMobileSticky disabledSkeleton isNotClosed boxOverContent--active')[0].style.display='none';")


def is_float(input):
    try:
        float(input)
        return True
    except ValueError:
        return False
