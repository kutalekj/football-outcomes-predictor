from utils import hide_sdk_banner
from selenium import webdriver
from selenium.webdriver.chrome.webdriver import WebDriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait as Wait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager

# Set webdriver
options = Options()
options.add_experimental_option("detach", True)
# options.add_argument("--headless")
options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.150 Safari/537.36")
driver: WebDriver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# Load main page
driver.get("https://www.tipsport.cz//")
driver.maximize_window()
# hide_sdk_banner(driver)

# "PŘIHLÁSIT SE"
Wait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, 'button[data-atid="headerLogin"]')))
login_button = driver.find_element(By.CSS_SELECTOR, 'button[data-atid="headerLogin"]')
login_button.click()
# driver.execute_script("arguments[0].click();", login_button)

# CREDENTIALS
Wait(driver, 10).until(EC.visibility_of_element_located((By.CSS_SELECTOR, '.ModalDialogstyled__ModalDialogTitleText-sc-byfvwo-0')))

email_field = driver.find_element(By.CSS_SELECTOR, 'input[data-atid="txt-username"]')
email_field.clear()
email_field.send_keys("jirka.kutalek@centrum.cz")

password_field = driver.find_element(By.CSS_SELECTOR, 'input[data-atid="txt-password"]')
password_field.clear()
password_field.send_keys("jnagano25")

Wait(driver, 3)

driver.quit()
