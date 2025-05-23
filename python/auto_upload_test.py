from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.action_chains import ActionChains
import openpyxl
from datetime import datetime
import time
import os
from selenium.webdriver.chrome.options import Options
import random

# 參數設定
APK_FILE = r"C:\\Users\\rich cheng\\Downloads\\app-debug.apk"  # APK檔案完整路徑（使用原始字串）
DIFFICULTY_VALUE = "2"    # 模擬難度對應 value，這裡是 Normal
MEMO_TEXT = "120normal自動化可悲嘗試"
MAX_SUCCESS = 5           # 要成功紀錄幾次後結束

# 建立 Excel
wb = openpyxl.Workbook()
ws = wb.active
ws.title = "Simulation Results"
ws.append(["時間", "APK 檔案", "難度", "Memo", "執行時間", "總分", "Accuracy", "Found Correct Item"])

# 建立瀏覽器與等待器
options = Options()
options.add_experimental_option("excludeSwitches", ["enable-automation"])
options.add_experimental_option('useAutomationExtension', False)
options.add_argument("--disable-blink-features=AutomationControlled")
driver = webdriver.Chrome(options=options)
driver.execute_cdp_cmd(
    "Page.addScriptToEvaluateOnNewDocument",
    {
        "source": """
        Object.defineProperty(navigator, 'webdriver', {get: () => undefined})
        """
    },
)
wait = WebDriverWait(driver, 30)

def login_and_go_simulation():
    driver.get("https://jaxa.krpc.jp/user-home")  # 登入頁網址
    
    # 等待帳號密碼輸入框出現
    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'input[placeholder="Account ID"]'))).send_keys("z20g3jeuqc")
    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'input[placeholder="Password"]'))).send_keys("hhr.20>,l_;'l/x")

    # 點擊登入按鈕
    login_button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, 'button.login-form-button')))
    login_button.click()

    # 等待 Simulation 連結出現並點擊
    simulation_link = wait.until(EC.element_to_be_clickable((By.XPATH, "//span[text()='Simulation']")))
    simulation_link.click()

    # 等待 simulation slot 狀態為 Available
    print("等待 slot 可用...")
    while True:
        slot_text = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div.slot-status"))).text
        print("目前 slot 狀態:", slot_text)
        if slot_text == "Available":
            break
        time.sleep(3)
    print("slot 可用，開始模擬...")

def upload_and_start():
    print("上傳 APK 檔案...")
    file_input = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'input.dz-hidden-input[type="file"]')))
    # 模擬人類點擊 input
    ActionChains(driver).move_to_element(file_input).pause(random.uniform(0.5, 1.2)).click().perform()
    time.sleep(random.uniform(0.5, 1.5))
    file_input.send_keys(APK_FILE)
    # 在每個步驟間加入隨機延遲
    time.sleep(random.uniform(0.5, 1.2))
    difficulty_dropdown = wait.until(EC.presence_of_element_located(
        (By.XPATH, "//div[contains(@class,'input-div--target') and .//span[text()='Simulator Level']]//select[@class='dropdown']")
    ))
    wait.until(lambda driver: len(Select(difficulty_dropdown).options) > 1)
    time.sleep(random.uniform(0.5, 1.2))
    select = Select(difficulty_dropdown)
    print("可選擇的難度:")
    for option in select.options:
        print(f"value={option.get_attribute('value')}, text={option.text}")
    select.select_by_value(DIFFICULTY_VALUE)
    time.sleep(random.uniform(0.5, 1.2))
    memo_input = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'input.standard-text-box')))
    memo_input.clear()
    memo_input.send_keys(MEMO_TEXT)
    time.sleep(random.uniform(0.5, 1.2))
    start_button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, 'button.button--start:not(.disabled)')))
    start_button.click()
    # 按下 Start 後等待並點擊確認對話框的 OK 按鈕（開始確認）
    # 等待按鈕出現後，先 sleep 0.5 秒再點擊
    # 等待 OK 按鈕內部 <span> 出現並點擊
    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'div.modal-container')))
    ok_button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, 'button.ok-button')))
    try:
        wait.until(EC.invisibility_of_element_located((By.CSS_SELECTOR, ".modal-overlay")))
    except:
        pass
    driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", ok_button)
    time.sleep(random.uniform(0.8, 1.5))
    ActionChains(driver).move_to_element(ok_button).pause(random.uniform(0.2, 0.7)).click().perform()
    try:
        time.sleep(random.uniform(1.5, 2.5))
        wait.until(EC.invisibility_of_element_located((By.XPATH, "//span[text()='Currently uploading the APK file']")))
    except Exception as e:
        print("Warning: Uploading message did not disappear as expected.", e)
    print("模擬已開始，等待結果...")



def wait_simulation_finished():
    while True:
        driver.refresh()
        time.sleep(5)  # 等待頁面刷新
        # 等待模擬狀態更新
        slot_status = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div.slot-status"))).text
        print(f"模擬狀態：{slot_status}")
        if slot_status == "Finished":
            return True
        elif slot_status != "In Progress":
            print("模擬狀態異常，重新上傳")
            return False
        time.sleep(60)

def parse_result_and_save_excel():
    view_button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "button.button--view")))
    view_button.click()

    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div.dashboard-page-status")))

    exec_time = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'span.status-value'))).text
    total_score = driver.find_element(By.XPATH, "//span[contains(text(),'Total Score')]/following-sibling::span").text
    accuracy = driver.find_element(By.XPATH, "//span[contains(text(),'Correct matching of area and item')]/following-sibling::span").text

    found_icon = driver.find_element(By.XPATH, "//span[contains(text(),'Found the correct item')]/following-sibling::span/i")
    found_correct_item = "Yes" if "fa-check-circle" in found_icon.get_attribute("class") else "No"

    ws.append([
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        os.path.basename(APK_FILE),
        DIFFICULTY_VALUE,
        MEMO_TEXT,
        exec_time,
        total_score,
        accuracy,
        found_correct_item
    ])
    wb.save("simulation_results.xlsx")
    print("結果已寫入 Excel")

def go_back_to_simulator():
    back_button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, 'a.back-button')))
    back_button.click()
    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'input.dz-hidden-input[type="file"]')))
    print("已回到模擬首頁")

def main():
    login_and_go_simulation()
    success_count = 0
    while success_count < MAX_SUCCESS:
        try:
            upload_and_start()
            if wait_simulation_finished():
                parse_result_and_save_excel()
                success_count += 1
                print(f"已成功完成 {success_count} 次模擬\n")
                go_back_to_simulator()
            else:
                print("模擬失敗，重新上傳 APK")
                time.sleep(5)
                
        except Exception as e:
            print(f"執行過程發生錯誤: {e}")
            break

    print("模擬流程結束")
    driver.quit()

if __name__ == "__main__":
    main()
