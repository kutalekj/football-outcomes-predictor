import time

from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait as Wait


def hide_sdk_banner(driver, sleep: float = 1.0, include_placeholder: bool = True):
    """
    Hides the Flashscore/OneTrust banner if it exists.
    Safe to call even if the banner is not present.
    """
    js = """
    const el = document.getElementById('onetrust-banner-sdk');
    if (el) el.style.display = 'none';
    """
    try:
        driver.execute_script(js)
    except Exception:
        # ignore any JS execution errors silently
        pass

    if include_placeholder:
        js2 = """
        const ph = document.getElementById('onetrust-pc-sdk');
        if (ph) ph.style.display = 'none';
        """
        try:
            driver.execute_script(js2)
        except Exception:
            pass

    import time

    time.sleep(sleep)


def hide_advert_banner(driver, sleep=2.0):
    time.sleep(sleep)
    driver.execute_script(
        "document.getElementsByClassName('boxOverContent boxOverContent--type-2 isSticky isMobile"
        "Sticky disabledSkeleton isNotClosed boxOverContent--active')[0].style.display='none';"
    )


def hide_tipsport_consent_banner(driver, sleep=2.0):
    time.sleep(sleep)
    driver.execute_script(
        """
        var banner = document.querySelector("[class^='Consentstyled__Banner']");
        if (banner) {
            banner.style.display = 'none';
        }
        """
    )


_cookie_handled = False  # process-wide flag


def dismiss_cookie_banner(driver, timeout_short=2):
    """Hide/dismiss cookie banner if present. No blocking waits."""
    global _cookie_handled
    if _cookie_handled:
        return

    try:
        # Fast presence check by ID
        if driver.execute_script("return !!document.getElementById('onetrust-banner-sdk');"):
            # Try button variations quickly
            for locator in [
                (By.ID, "onetrust-accept-btn-handler"),
                (
                    By.CSS_SELECTOR,
                    "#onetrust-accept-btn-handler, #onetrust-reject-all-handler, .onetrust-close-btn-handler",
                ),
                (By.XPATH, "//button[contains(., 'Accept')]"),
            ]:
                try:
                    btn = Wait(driver, timeout_short).until(EC.element_to_be_clickable(locator))
                    btn.click()
                    _cookie_handled = True
                    return
                except TimeoutException:
                    continue
            # As a last resort: hide the banner via JS and mark handled
            driver.execute_script(
                """
                var b=document.getElementById('onetrust-banner-sdk');
                if (b) { b.style.display='none'; }
            """
            )
            _cookie_handled = True
    except Exception:
        # Never block scraping because of the banner
        pass


def is_float(input):
    try:
        float(input)
        return True
    except ValueError:
        return False
