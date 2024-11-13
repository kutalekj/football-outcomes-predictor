import os
import time
import datetime
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import ElementClickInterceptedException
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from webdriver_manager.chrome import ChromeDriverManager


def setup_driver():
    chrome_options = Options()
    # chrome_options.add_argument('--headless')  # Run headless browser
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument("--window-size=1920,1080")  # Ensure all content is loaded
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--disable-dev-shm-usage")

    # Manually specify the path to chromedriver.exe
    driver_path = r'C:\Users\kutalekj\.wdm\drivers\chromedriver\win64\130.0.6723.117\chromedriver-win32\chromedriver.exe'

    service = Service(executable_path=driver_path)

    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver


def generate_date_list(start_date, end_date):
    date_list = []
    current_date = start_date
    while current_date <= end_date:
        date_list.append(current_date)
        current_date += datetime.timedelta(days=1)
    return date_list


def navigate_to_date(driver, date):
    # Open the homepage
    driver.get('https://makeyourstats.com/?date=2023-11-12')
    wait = WebDriverWait(driver, 10)

    # Wait for the date picker button to be clickable and enabled
    calendar_button = wait.until(
        EC.element_to_be_clickable((By.CSS_SELECTOR, 'button.nav-link.d-flex.btn-outline-primary-live.text-primary'))
    )

    # Ensure the button is enabled
    while 'disabled' in calendar_button.get_attribute('class'):
        print("Waiting for date picker button to become enabled...")
        time.sleep(1)
        calendar_button = driver.find_element(By.CSS_SELECTOR,
                                              'button.nav-link.d-flex.btn-outline-primary-live.text-primary')

    # Proceed to click the button
    try:
        calendar_button.click()
        print("Date picker opened successfully.")
    except Exception as e:
        print(f"Error clicking date picker button: {type(e).__name__}: {e}")
        return

    # Select the date
    select_date_in_calendar(driver, date)

    # Wait for fixtures to load
    time.sleep(2)

    # Scroll to the bottom to load all fixtures
    scroll_to_load_all_fixtures(driver)


def select_date_in_calendar(driver, date):
    wait = WebDriverWait(driver, 10)

    # Wait for the calendar modal to appear
    wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, 'div.vc-container')))

    # Navigate to the correct month and year
    while True:
        # Get the displayed month and year
        month_year_element = driver.find_element(By.CSS_SELECTOR, 'div.vc-title')
        displayed_month_year = month_year_element.text.strip()
        target_month_year = date.strftime('%B %Y')

        if displayed_month_year == target_month_year:
            break
        else:
            # Click the previous or next month button
            displayed_date = datetime.datetime.strptime(displayed_month_year, '%B %Y').date()
            if date < displayed_date:
                prev_button = driver.find_element(By.CSS_SELECTOR, 'div.vc-arrow.is-left')
                prev_button.click()
            else:
                next_button = driver.find_element(By.CSS_SELECTOR, 'div.vc-arrow.is-right')
                if 'is-disabled' in next_button.get_attribute('class'):
                    print("Cannot navigate to future months.")
                    return
                next_button.click()
            time.sleep(0.5)

    # Click on the correct day
    day_elements = driver.find_elements(By.CSS_SELECTOR, 'div.vc-day')
    for day_element in day_elements:
        # Get the aria-label attribute
        span_element = day_element.find_element(By.CSS_SELECTOR, 'span.vc-day-content')
        aria_label = span_element.get_attribute('aria-label')
        day_date = datetime.datetime.strptime(aria_label, '%A, %B %d, %Y')
        if day_date.date() == date:
            # Check if the day is not disabled
            is_disabled = span_element.get_attribute('aria-disabled')
            if is_disabled == 'false':
                span_element.click()
                print(f"Date {date} selected.")
                time.sleep(1)
                return
            else:
                print(f"Date {date} is disabled and cannot be selected.")
                return
    print(f"Date {date} not found in the calendar.")


def close_overlays(driver):
    wait = WebDriverWait(driver, 10)
    try:
        overlay = wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, 'div.overlay-selector')))  # Replace with actual selector
        close_button = overlay.find_element(By.CSS_SELECTOR, 'button.close')  # Adjust selector as needed
        close_button.click()
        print("Closed overlay.")
    except:
        pass  # No overlay found


def scroll_to_load_all_fixtures(driver):
    last_height = driver.execute_script("return document.body.scrollHeight")
    while True:
        # Scroll down to bottom
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        # Wait to load page
        time.sleep(2)
        # Calculate new scroll height and compare with last scroll height
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height


def get_laliga2_fixtures(driver):
    wait = WebDriverWait(driver, 10)
    fixture_elements = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'div.card.m-1.py-1')))

    laliga2_fixtures = []
    for fixture in fixture_elements:
        try:
            # Find the competition name
            competition_element = fixture.find_element(By.CSS_SELECTOR, 'p.league-name')
            competition_name = competition_element.text.strip()
            if 'La Liga 2' in competition_name or 'LaLiga2' in competition_name or 'Segunda División' in competition_name:
                laliga2_fixtures.append(fixture)
        except Exception as e:
            continue
    return laliga2_fixtures


def extract_xg_data(driver, fixture):
    wait = WebDriverWait(driver, 10)

    # Click on the fixture to open match details
    try:
        clickable_element = fixture.find_element(By.CSS_SELECTOR, 'p.text-primary.pointer')
        driver.execute_script("arguments[0].scrollIntoView();", clickable_element)
        clickable_element.click()
    except Exception as e:
        print(f"Error clicking on fixture: {e}")
        return None

    # Wait for the match details to load
    time.sleep(2)

    try:
        # Get team names
        team_name_elements = driver.find_elements(By.CSS_SELECTOR, 'p.font-weight-bolder.text-center.mt-1.text-dark')
        home_team = team_name_elements[0].text.strip()
        away_team = team_name_elements[1].text.strip()

        # Find the rows containing xG data
        rows = driver.find_elements(By.CSS_SELECTOR, 'div.d-flex.justify-content-around.border-bottom.pt-1')

        xg_value_home = None
        xg_value_away = None
        xga_value_home = None
        xga_value_away = None

        for row in rows:
            # Get all columns
            columns = row.find_elements(By.CSS_SELECTOR, 'div.pb-1')
            if len(columns) != 3:
                continue  # Skip if not three columns
            left_value_element = columns[0].find_element(By.CSS_SELECTOR, 'div.parent > span, div.parent > p')
            middle_text_element = columns[1].find_element(By.CSS_SELECTOR, 'p.text-center')
            right_value_element = columns[2].find_element(By.CSS_SELECTOR, 'div.parent > span, div.parent > p')

            left_value = left_value_element.text.strip()
            middle_text = middle_text_element.text.strip()
            right_value = right_value_element.text.strip()

            if middle_text == 'Expected goals (xG)':
                xg_value_home = left_value
                xg_value_away = right_value
            elif middle_text == 'Expected goals against (xGA)':
                xga_value_home = left_value
                xga_value_away = right_value

        # Close the match details
        close_button = driver.find_element(By.CSS_SELECTOR, 'button.close')
        close_button.click()

        return {
            'home_team': home_team,
            'away_team': away_team,
            'home_xg': xg_value_home,
            'away_xg': xg_value_away,
            'home_xga': xga_value_home,
            'away_xga': xga_value_away
        }
    except Exception as e:
        print(f"Error extracting xG data: {e}")
        # Close the match details
        try:
            close_button = driver.find_element(By.CSS_SELECTOR, 'button.close')
            close_button.click()
        except:
            pass
        return None


def scrape_xg_data():
    driver = setup_driver()
    start_date = datetime.date(2023, 11, 9)  # TODO: Adjust
    end_date = datetime.date(2023, 11, 9)
    date_list = generate_date_list(start_date, end_date)

    for date in date_list:
        try:
            print(f"Processing date: {date}")
            navigate_to_date(driver, date)
            laliga2_fixtures = get_laliga2_fixtures(driver)
            print(f"Found {len(laliga2_fixtures)} LaLiga2 fixtures on {date}")

            data = []
            for fixture in laliga2_fixtures:
                xg_data = extract_xg_data(driver, fixture)
                if xg_data:
                    xg_data['date'] = date.strftime('%Y-%m-%d')
                    data.append(xg_data)

            # Save data to CSV
            if data:
                df = pd.DataFrame(data)
                # Save to CSV file per month
                month_str = date.strftime('%Y-%m')
                csv_filename = f'laliga2_xg_{month_str}.csv'
                if not os.path.exists(csv_filename):
                    df.to_csv(csv_filename, index=False)
                else:
                    df.to_csv(csv_filename, mode='a', header=False, index=False)
            else:
                print(f"No LaLiga2 matches on {date}")
        except Exception as e:
            print(f"Error processing date {date}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
    driver.quit()


if __name__ == "__main__":
    scrape_xg_data()
