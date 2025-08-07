import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from collections import defaultdict
from tqdm import tqdm
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

# Setup headless browser
options = webdriver.ChromeOptions()
options.add_argument('--headless')
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# Load BMU dropdown page
url = 'https://www.netareports.com/data/elexon/bmu.jsp'
driver.get(url)
time.sleep(3)

# Get all <option> elements from the dropdown
options = driver.find_elements(By.TAG_NAME, 'option')
bmu_links = []

for option in options:
    val = option.get_attribute('value')
    if val and 'id=' in val:
        bmu_id = val.split('id=')[-1]
        bmu_name = option.text.strip()
        bmu_links.append((bmu_id, bmu_name))

print(f"Found {len(bmu_links)} BMUs")

# Scrape detail for each BMU
bmu_data = []

for i, (bmu_id, bmu_name) in enumerate(tqdm(bmu_links, desc="Scraping BMUs", unit="BMU")):
    try:
        driver.get(f"https://www.netareports.com/data/elexon/bmu.jsp?id={bmu_id}")

        # Wait up to 1.5 seconds for the <table> to load
        WebDriverWait(driver, 1.5).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "table tr"))
        )

        rows = driver.find_elements(By.CSS_SELECTOR, 'table tr')
        row_data = defaultdict(str)
        row_data['BMU_ID'] = bmu_id
        row_data['BMU_Name'] = bmu_name

        for row in rows:
            tds = row.find_elements(By.TAG_NAME, 'td')
            if not tds or len(tds) < 2:
                continue
            key = tds[0].text.strip().replace(":", "")
            values = [td.text.strip() for td in tds[1:]]
            row_data[key] = ' | '.join(v for v in values if v)

        bmu_data.append(row_data)

        # ✅ Option 2: Save partial progress every 100 BMUs
        if i > 0 and i % 100 == 0:
            pd.DataFrame(bmu_data).to_csv("bmu_detailed_info_partial.csv", index=False)
            print(f"Progress saved at BMU {i}")

    except TimeoutException:
        print(f"Timeout loading BMU {bmu_id} — skipping...")
        continue

    except Exception as e:
        # ✅ Option 1: Catch all unexpected errors to prevent crashing
        print(f"Error loading BMU {bmu_id}: {e}")
        continue

driver.quit()

# Final save
df = pd.DataFrame(bmu_data)
df.to_csv("bmu_detailed_info.csv", index=False)
print("Saved as bmu_detailed_info.csv")