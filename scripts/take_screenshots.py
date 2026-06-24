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
from selenium.webdriver.common.keys import Keys

ROOT = Path(__file__).resolve().parents[1]

# Start Streamlit in a subprocess with cleared Spotify credentials to prevent slow external API requests
print("Starting Streamlit app...")
env = os.environ.copy()
env["SPOTIFY_CLIENT_ID"] = ""
env["SPOTIFY_CLIENT_SECRET"] = ""
streamlit_proc = subprocess.Popen(
    [sys.executable, "-m", "streamlit", "run", str(ROOT / "app" / "main.py"), "--server.port=8501", "--server.headless=true"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    cwd=str(ROOT),
    env=env,
)

# Wait for Streamlit to start
print("Waiting for Streamlit server to spin up...")
time.sleep(50)

# Setup Chrome headless
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("--window-size=1920,1080")
chrome_options.add_argument("--force-device-scale-factor=1.25")

try:
    from webdriver_manager.chrome import ChromeDriverManager
    service = Service(ChromeDriverManager().install())
except Exception:
    service = None

driver = webdriver.Chrome(options=chrome_options, service=service)

screenshots_dir = ROOT / "screenshots"
screenshots_dir.mkdir(exist_ok=True)

try:
    # Navigate to the app
    print("Navigating to app...")
    driver.get("http://localhost:8501")

    # Wait until empty state is rendered (which means indexing is finished and UI is ready)
    print("Waiting for index compilation and empty state to render...")
    wait = WebDriverWait(driver, 120)
    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, ".empty-state")))
    time.sleep(3)

    # Screenshot 1: Empty state (default page)
    print("Taking screenshot 1: Empty state...")
    driver.save_screenshot(str(screenshots_dir / "01_empty_state.png"))
    print("  Saved 01_empty_state.png")

    # Screenshot 2: Vibe search - "chill night drive"
    print("Taking screenshot 2: Vibe search...")
    # Find and click the "Chill Night Drive" button in the sidebar
    vibe_btn = None
    buttons = driver.find_elements(By.TAG_NAME, "button")
    for btn in buttons:
        if "chill night drive" in btn.text.lower():
            vibe_btn = btn
            break
    if vibe_btn:
        vibe_btn.click()
        print("  Clicked Chill Night Drive vibe button")
    else:
        raise Exception("Vibe button 'Chill Night Drive' not found")

    # Wait for results cards to render
    print("Waiting for recommendations results to load...")
    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, ".card")))
    time.sleep(3)

    # Click the "Why?" button on the first card to show the explanation drawer
    print("  Expanding explanation for the first recommendation card...")
    why_btns = driver.find_elements(By.CSS_SELECTOR, ".why-btn")
    if why_btns:
        why_btns[0].click()
        time.sleep(1)

    # Click the play button on the first card to activate player
    print("  Clicking the play button on the first card...")
    play_circles = driver.find_elements(By.CSS_SELECTOR, ".play-circle")
    if play_circles:
        play_circles[0].click()
        time.sleep(2)

    driver.save_screenshot(str(screenshots_dir / "02_vibe_search.png"))
    print("  Saved 02_vibe_search.png")

    # Screenshot 3: Text search - "Blinding Lights"
    print("Taking screenshot 3: Text search...")
    
    # Locate search input
    inp = driver.find_element(By.CSS_SELECTOR, "section[data-testid='stSidebar'] input")
    inp.send_keys(Keys.CONTROL + "a")
    inp.send_keys(Keys.BACKSPACE)
    inp.send_keys("Blinding Lights")
    time.sleep(1)

    # Click the "Get recommendations" button
    rec_btn = None
    for btn in driver.find_elements(By.TAG_NAME, "button"):
        if "get recommendations" in btn.text.lower():
            rec_btn = btn
            break
    if rec_btn:
        rec_btn.click()
        print("  Clicked Get recommendations button")
    else:
        inp.send_keys(Keys.ENTER)
        print("  Pressed Enter in input")

    # Wait for results to load
    print("Waiting for search results to load...")
    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, ".card")))
    time.sleep(3)

    # Click "Why?" button on first result card
    print("  Expanding explanation for the search result...")
    why_btns = driver.find_elements(By.CSS_SELECTOR, ".why-btn")
    if why_btns:
        why_btns[0].click()
        time.sleep(1)

    driver.save_screenshot(str(screenshots_dir / "03_text_search.png"))
    print("  Saved 03_text_search.png")

    print(f"\nAll high-fidelity screenshots saved to {screenshots_dir}/")

except Exception as e:
    print(f"Error: {e}")
    try:
        driver.save_screenshot(str(screenshots_dir / "error.png"))
        print("  Saved error.png screenshot for debugging")
    except Exception as se:
        print(f"Failed to save error screenshot: {se}")
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
