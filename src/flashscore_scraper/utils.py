import time

from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait as Wait


def hide_sdk_banner(driver, sleep: float = 0.5, include_placeholder: bool = True):
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
    """
    Hide/dismiss OneTrust cookie banner if present.
    Always checks the DOM; safe to call many times.
    """
    try:
        # Is the banner present & displayed?
        is_present = driver.execute_script(
            """
            var el = document.getElementById('onetrust-banner-sdk');
            return !!(el && el.offsetParent !== null);
        """
        )
        if not is_present:
            return

        # Try the standard buttons quickly
        for locator in [
            (By.ID, "onetrust-accept-btn-handler"),
            (By.ID, "onetrust-reject-all-handler"),
            (By.CSS_SELECTOR, ".onetrust-close-btn-handler"),
        ]:
            try:
                btn = Wait(driver, timeout_short).until(EC.element_to_be_clickable(locator))
                btn.click()
                # brief settle
                time.sleep(0.2)
                break
            except TimeoutException:
                continue

        # If banner still visible, hard-hide it
        driver.execute_script(
            """
            var b=document.getElementById('onetrust-banner-sdk');
            if (b) { b.style.display='none'; }
            var pc=document.getElementById('onetrust-pc-sdk');
            if (pc) { pc.style.display='none'; }
        """
        )
    except Exception:
        # Never block scraping because of the banner
        pass


def is_float(input):
    try:
        float(input)
        return True
    except ValueError:
        return False
