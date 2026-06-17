"""Take screenshots of the Moodify Streamlit app using Selenium + Chrome headless."""

import subprocess
import sys
import time
import os
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

ROOT = Path(__file__).resolve().parents[1]

# Start Streamlit in a subprocess
print("Starting Streamlit app...")
streamlit_proc = subprocess.Popen(
    [sys.executable, "-m", "streamlit", "run", str(ROOT / "app" / "main.py"), "--server.port=8501", "--server.headless=true"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    cwd=str(ROOT),
)

# Wait for Streamlit to start (need longer for FAISS + TF-IDF index building)
print("Waiting 45s for Streamlit to fully load indices...")
time.sleep(45)

# Setup Chrome headless
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("--window-size=1920,1080")
chrome_options.add_argument("--force-device-scale-factor=1.5")

try:
    from webdriver_manager.chrome import ChromeDriverManager
    service = Service(ChromeDriverManager().install())
except Exception:
    service = None

driver = webdriver.Chrome(options=chrome_options, service=service)

screenshots_dir = ROOT / "screenshots"
screenshots_dir.mkdir(exist_ok=True)

try:
    # Navigate to the app and wait for spinner to disappear
    print("Navigating to app...")
    driver.get("http://localhost:8501")

    # Wait until "Building smart search index..." spinner disappears
    print("Waiting for smart search index to finish building...")
    wait = WebDriverWait(driver, 60)
    # Wait for the spinner to disappear by checking for the hero text
    wait.until(EC.presence_of_element_located((By.XPATH, "//*[contains(text(), 'Moodify')]")))
    # Extra wait for any remaining spinners
    time.sleep(5)

    # Screenshot 1: Empty state (default page)
    print("Taking screenshot 1: Empty state...")
    driver.save_screenshot(str(screenshots_dir / "01_empty_state.png"))
    print("  Saved 01_empty_state.png")

    # Screenshot 2: Vibe search - "chill night drive"
    print("Taking screenshot 2: Vibe search...")
    # Find text input by placeholder
    inputs = driver.find_elements(By.TAG_NAME, "input")
    for inp in inputs:
        try:
            placeholder = inp.get_attribute("placeholder") or ""
            if "vibe" in placeholder.lower() or inp.is_displayed():
                inp.clear()
                inp.send_keys("chill night drive")
                break
        except Exception:
            continue

    time.sleep(1)
    # Click "Get recommendations" button
    buttons = driver.find_elements(By.TAG_NAME, "button")
    for btn in buttons:
        if "recommendations" in btn.text.lower() or "get" in btn.text.lower():
            try:
                btn.click()
                break
            except Exception:
                continue

    # Wait for results to load
    time.sleep(8)
    driver.save_screenshot(str(screenshots_dir / "02_vibe_search.png"))
    print("  Saved 02_vibe_search.png")

    # Screenshot 3: Text search - "Blinding Lights"
    print("Taking screenshot 3: Text search...")
    driver.get("http://localhost:8501")
    # Wait for spinner again
    time.sleep(25)
    wait.until(EC.presence_of_element_located((By.XPATH, "//*[contains(text(), 'Moodify')]")))
    time.sleep(3)

    for inp in driver.find_elements(By.TAG_NAME, "input"):
        try:
            placeholder = inp.get_attribute("placeholder") or ""
            if "vibe" in placeholder.lower() or inp.is_displayed():
                inp.clear()
                inp.send_keys("Blinding Lights")
                break
        except Exception:
            continue

    time.sleep(1)
    for btn in driver.find_elements(By.TAG_NAME, "button"):
        if "recommendations" in btn.text.lower() or "get" in btn.text.lower():
            try:
                btn.click()
                break
            except Exception:
                continue

    time.sleep(8)
    driver.save_screenshot(str(screenshots_dir / "03_text_search.png"))
    print("  Saved 03_text_search.png")

    print(f"\nAll screenshots saved to {screenshots_dir}/")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

finally:
    driver.quit()
    streamlit_proc.terminate()
    try:
        streamlit_proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        streamlit_proc.kill()
    print("Streamlit stopped.")
