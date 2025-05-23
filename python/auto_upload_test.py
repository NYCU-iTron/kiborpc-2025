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
from selenium.common.exceptions import TimeoutException, StaleElementReferenceException, NoSuchElementException

# 參數設定
APK_FILE = r"C:\\Users\\Jonat\\Downloads\\v1.3.0.apk"  # APK檔案完整路徑（使用原始字串）
DIFFICULTY_VALUE = "2"    # 模擬難度對應 value，這裡是 Normal
MEMO_TEXT = "v1.3.0 normal自動化嘗試"
MAX_SUCCESS = 6           # 要成功紀錄幾次後結束

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
    
    # 確保模擬頁面已載入，等待任一 slot status 元件出現即可
    print("已導航至模擬頁面，等待頁面元素載入...")
    try:
        wait.until(EC.presence_of_element_located((By.XPATH, "//div[@class='slot']"))) # Wait for a slot container
        print("模擬頁面已準備就緒。")
    except TimeoutException:
        print("錯誤: 等待模擬頁面主要內容 (//div[@class='slot']) 載入超時。")
        driver.quit()
        raise

def parse_result_and_save_excel(current_slot_panel_for_results):
    view_button_selector_css = "div.button-container button.button--view:not(.disabled)"
    max_click_attempts = 3
    attempt = 0
    clicked_successfully = False

    while attempt < max_click_attempts and not clicked_successfully:
        attempt += 1
        print(f"DEBUG (Attempt {attempt}/{max_click_attempts}): Looking for and trying to click 'View' button inside the finished slot panel...")
        try:
            # Always re-locate the button in each attempt within the current_slot_panel_for_results context
            view_button = WebDriverWait(current_slot_panel_for_results, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, view_button_selector_css))
            )
            
            print(f"DEBUG (Attempt {attempt}): 'View' button found and clickable. Scrolling and attempting click.")
            # 將元素的頂部滾動到視口的頂部，確保它不被上方元素遮擋
            driver.execute_script("arguments[0].scrollIntoView({block: 'start', inline: 'nearest'});", view_button)
            time.sleep(random.uniform(0.3, 0.7)) # Short pause after scroll
            
            # Direct click on the freshly located button
            view_button.click()
            clicked_successfully = True
            print(f"DEBUG (Attempt {attempt}): 'View' button clicked successfully.")

        except StaleElementReferenceException as e_stale:
            print(f"DEBUG (Attempt {attempt}): StaleElementReferenceException encountered when trying to click 'View' button. Details: {e_stale}")
            if attempt >= max_click_attempts:
                print(f"錯誤: 連續 {max_click_attempts} 次嘗試點擊 'View' 按鈕失敗 (StaleElementReferenceException)。")
                # Optional: print panel HTML for debugging
                # print(f"DEBUG: Finished slot panel HTML (first 500 chars): {current_slot_panel_for_results.get_attribute('innerHTML')[:500]}")
                return False
            print("Retrying after a short pause...")
            time.sleep(1) # Wait 1 second before retrying
        except TimeoutException:
            print(f"錯誤 (Attempt {attempt}): 在已完成的槽面板内等待 'View' 按鈕 (selector: {view_button_selector_css}) 可用或可點擊超時.")
            # Optional: print panel HTML for debugging
            # print(f"DEBUG: Finished slot panel HTML (first 500 chars): {current_slot_panel_for_results.get_attribute('innerHTML')[:500]}")
            return False # If timeout, probably no point in retrying immediately
        except Exception as e_general_click:
            print(f"錯誤 (Attempt {attempt}): 點擊 'View' 按鈕時發生未預期錯誤: {e_general_click}")
            # Optional: print panel HTML for debugging
            # print(f"DEBUG: Finished slot panel HTML (first 500 chars): {current_slot_panel_for_results.get_attribute('innerHTML')[:500]}")
            return False # General error, stop trying

    if not clicked_successfully:
        # This case should ideally be caught by the StaleElementReferenceException handling if max_attempts is reached
        print(f"錯誤: 未能成功點擊 'View' 按鈕，即使在 {max_click_attempts} 次嘗試後。")
        return False

    try:
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div.dashboard-page-status")))
        print("DEBUG: Results page loaded.")
    except TimeoutException:
        print("錯誤: 等待結果頁面 (div.dashboard-page-status) 載入超時。")
        # Try to go back if result page didn't load, to avoid getting stuck
        try:
            go_back_button_on_error = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, 'a.back-button')))
            go_back_button_on_error.click()
            print("DEBUG: Navigated back to simulator page after result page load timeout.")
        except Exception as e_back:
            print(f"DEBUG: Could not navigate back after result page load timeout: {e_back}")
        return False

    try:
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
        print(f"結果已成功寫入 Excel: APK='{os.path.basename(APK_FILE)}', Score='{total_score}'")
        return True
    except Exception as e:
        print(f"解析結果或寫入Excel時發生錯誤: {e}")
        # Attempt to go back to simulator page to prevent getting stuck on result page
        try:
            go_back_button_on_parse_error = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, 'a.back-button')))
            go_back_button_on_parse_error.click()
            print("DEBUG: Navigated back to simulator page after parsing/Excel error.")
        except Exception as e_back_parse:
            print(f"DEBUG: Could not navigate back after parsing/Excel error: {e_back_parse}")
        return False

def go_back_to_simulator():
    try:
        print("DEBUG: Attempting to go back to simulator main page...")
        back_button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, 'a.back-button')))
        driver.execute_script("arguments[0].scrollIntoView({block: 'center', inline: 'nearest'});", back_button)
        time.sleep(random.uniform(0.2,0.5))
        back_button.click()
        # Wait for an element that signifies the simulation page is loaded
        wait.until(EC.presence_of_element_located((By.XPATH, "//div[@class='slot']")))
        print("已成功回到模擬首頁。")
        time.sleep(random.uniform(1,2)) # Give page a moment to settle
    except Exception as e:
        print(f"返回模擬首頁時發生錯誤: {e}. 嘗試刷新頁面作為後備方案。")
        driver.refresh()
        try:
            wait.until(EC.presence_of_element_located((By.XPATH, "//div[@class='slot']")))
            print("頁面刷新後，已在模擬首頁。")
        except Exception as e_refresh:
            print(f"刷新頁面後仍無法確認回到模擬首頁: {e_refresh}. 流程可能無法繼續。")
            raise # Re-raise if refresh also fails to get us to a good state

def main():
    login_and_go_simulation()
    success_count = 0
    
    # Ensure this selector is robust for your page structure
    slot_panel_xpath = "//div[@class='slot']"
    # This assumes the status div is a direct child or specific descendant relative to the slot panel
    # Needs to be relative to the slot panel if we iterate through panels and get status from each
    # For now, we'll get all status divs and assume their order matches slot panels.

    max_scan_retries_with_no_action = 3 # Number of full scans with no action before a longer pause/refresh
    scans_with_no_action_count = 0

    while success_count < MAX_SUCCESS:
        print("DEBUG: 循環開始，刷新頁面以獲取最新狀態...")
        driver.refresh()
        try:
            # Wait for the main slot panel container to be present after refresh
            WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.XPATH, slot_panel_xpath)))
            print("DEBUG: 頁面刷新完成，模擬面板已載入。")
            time.sleep(random.uniform(1, 2)) # Brief pause for all elements to settle
        except TimeoutException:
            print("錯誤：刷新後等待模擬面板載入超時。可能影響本輪掃描。嘗試繼續...")
            # For now, let it try to proceed, subsequent find_elements might fail gracefully or timeout
            # Consider adding a counter here to break if it happens too many times consecutively.
            pass # Allow the loop to continue and attempt to find elements

        print(f"--- 開始掃描週期 (已成功 {success_count}/{MAX_SUCCESS} 次) ---")
        action_taken_in_current_scan = False
        at_least_one_slot_is_available = False

        all_slot_interaction_panels = []
        # panel_info_list will store dicts: {'panel': WebElement, 'status': str, 'original_index': int}
        panel_info_list = [] 
        uploadable_slots_ordered = [] # List of panel WebElements that are "Available"
        all_global_file_inputs = []

        try:
            print("DEBUG: 重新獲取所有槽面板、狀態和全域文件輸入點...")
            WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.XPATH, slot_panel_xpath)))
            
            all_slot_interaction_panels = driver.find_elements(By.XPATH, slot_panel_xpath)
            all_global_file_inputs = driver.find_elements(By.CSS_SELECTOR, 'input.dz-hidden-input[type="file"]')

            if not all_slot_interaction_panels:
                print("警告: 未找到任何槽面板。等待10秒後刷新重試...")
                time.sleep(10)
                driver.refresh()
                continue

            print(f"DEBUG: 偵測到 {len(all_slot_interaction_panels)} 個槽面板。")
            print(f"DEBUG: 全域偵測到 {len(all_global_file_inputs)} 個 'input.dz-hidden-input[type=\"file\"]' 檔案輸入點。")

            # Populate panel_info_list and uploadable_slots_ordered
            for i, panel_element in enumerate(all_slot_interaction_panels):
                status_text = "Unknown"
                try:
                    status_text = panel_element.find_element(By.CSS_SELECTOR, "div.slot-status").text
                except NoSuchElementException:
                    print(f"警告: 槽面板 {i} 內未找到 div.slot-status。將此槽狀態視為未知。")
                panel_info_list.append({'panel': panel_element, 'status': status_text, 'original_index': i})
                if status_text == "Available":
                    uploadable_slots_ordered.append(panel_element)
                    at_least_one_slot_is_available = True # Set this flag if any slot is Available
            
            print(f"DEBUG: 偵測到 {len(uploadable_slots_ordered)} 個 'Available' 狀態的槽面板。")

            # Crucial Check: Number of "Available" slots vs. number of global file inputs
            if len(uploadable_slots_ordered) != len(all_global_file_inputs):
                print(f"警告: 'Available' 狀態槽 ({len(uploadable_slots_ordered)}) 與全域檔案輸入點 ({len(all_global_file_inputs)}) 數量不匹配。可能Dropzone狀態與UI不一致。")
                print(f"DEBUG: 所有槽面板狀態: {[info['status'] for info in panel_info_list]}")
                print("等待10秒後刷新重試...")
                time.sleep(10)
                driver.refresh()
                continue

        except TimeoutException:
            print("獲取槽面板/狀態/文件輸入點時超時。10秒後刷新重試...")
            time.sleep(10)
            driver.refresh()
            continue
        except Exception as e_fetch:
            print(f"獲取槽面板/狀態時發生一般錯誤: {e_fetch}。10秒後刷新重試...")
            time.sleep(10)
            driver.refresh()
            continue
        
        num_slots_to_process = len(all_slot_interaction_panels) # Should be same as len(panel_info_list)
        current_global_input_index = 0 # Counter for all_global_file_inputs

        for slot_info in panel_info_list:
            # current_panel_from_list = slot_info['panel'] # This might be stale
            current_status_text = slot_info['status']
            slot_index = slot_info['original_index']
            active_file_input = None # Reset for each slot, determined if 'Available'

            print(f"\n檢查槽 {slot_index}: 狀態 = '{current_status_text}'")

            if current_status_text == "Finished":
                print(f"槽 {slot_index} 已完成。準備解析結果...")
                panel_to_process = None
                try:
                    print(f"DEBUG: Re-fetching panel for slot_index {slot_index} (Finished) before scrolling and parsing.")
                    # Re-fetch all panels and select the one by original_index for maximum freshness
                    all_current_panels_on_page = driver.find_elements(By.XPATH, slot_panel_xpath)
                    if slot_index < len(all_current_panels_on_page):
                        panel_to_process = all_current_panels_on_page[slot_index]
                    else:
                        print(f"錯誤: 嘗試重新獲取槽面板 {slot_index} (Finished) 失敗 (索引 {slot_index} 超出範圍 {len(all_current_panels_on_page)})。跳過此槽。")
                        action_taken_in_current_scan = True # Count as action to avoid immediate long wait
                        continue # Skip to next slot_info

                    driver.execute_script("arguments[0].scrollIntoView({block: 'center', inline: 'nearest'});", panel_to_process)
                    time.sleep(0.5) # Pause after scroll

                    if parse_result_and_save_excel(panel_to_process):
                        success_count += 1
                        print(f"槽 {slot_index} 結果已成功處理。總成功次數: {success_count}/{MAX_SUCCESS}")
                    else:
                        print(f"槽 {slot_index} 結果處理失敗或跳過。")
                    
                    action_taken_in_current_scan = True
                    if success_count >= MAX_SUCCESS: break
                    
                    go_back_to_simulator() 
                    break # Break from for-loop to re-evaluate all slots from main while-loop
                
                except StaleElementReferenceException as e_stale_finished:
                    print(f"錯誤: 在為槽 {slot_index} (Finished) 處理面板時發生 StaleElementReferenceException: {e_stale_finished}")
                    print("將刷新並在下一輪嘗試。")
                    action_taken_in_current_scan = True # To trigger a quick rescan after refresh
                    # No `go_back_to_simulator` here, let the main loop refresh and retry.
                    break # Break to main while loop to refresh and rescan
                except Exception as e_refetch_finished:
                    print(f"錯誤: 在為槽 {slot_index} (Finished) 重新獲取或處理面板時發生未預期異常: {e_refetch_finished}")
                    action_taken_in_current_scan = True
                    # No `go_back_to_simulator` here, let the main loop refresh and retry.
                    break # Break to main while loop to refresh and rescan

            elif current_status_text == "Available":
                panel_to_process_available = None # Panel for 'Available' slot
                # This slot is "Available". It should correspond to the next global file input.
                if current_global_input_index < len(all_global_file_inputs):
                    active_file_input = all_global_file_inputs[current_global_input_index]
                    print(f"DEBUG: 槽 {slot_index} ('Available') 對應到全域檔案輸入點索引 {current_global_input_index}.")
                else:
                    print(f"錯誤: 槽 {slot_index} 狀態 'Available' 但預期檔案輸入點索引 ({current_global_input_index}) 超出範圍。跳過此槽。")
                    action_taken_in_current_scan = True # Avoid immediate long wait
                    continue

                print(f"槽 {slot_index} 可用。開始上傳並配置模擬...")
                try:
                    print(f"DEBUG: Re-fetching panel for slot_index {slot_index} (Available) before interaction.")
                    all_current_panels_on_page_avail = driver.find_elements(By.XPATH, slot_panel_xpath)
                    if slot_index < len(all_current_panels_on_page_avail):
                        panel_to_process_available = all_current_panels_on_page_avail[slot_index]
                    else:
                        print(f"錯誤: 嘗試重新獲取槽面板 {slot_index} (Available) 失敗 (索引 {slot_index} 超出範圍 {len(all_current_panels_on_page_avail)})。跳過此槽。")
                        action_taken_in_current_scan = True
                        current_global_input_index += 1 # Consume the input mapping attempt
                        continue

                    driver.execute_script("arguments[0].scrollIntoView({block: 'center', inline: 'nearest'});", panel_to_process_available)
                    time.sleep(0.5)

                    print(f"DEBUG: 上傳 APK 檔案 '{os.path.basename(APK_FILE)}' 到槽 {slot_index}...")
                    active_file_input.send_keys(APK_FILE)
                    print(f"DEBUG: 檔案已傳送至槽 {slot_index}")
                    time.sleep(random.uniform(0.8, 1.5))

                    print(f"DEBUG: 在槽位 {slot_index} 內尋找 Simulator Level 的父容器 div.input-div--target...")
                    difficulty_container_xpath = ".//div[contains(@class,'input-div--target') and .//span[text()='Simulator Level']]"
                    difficulty_container = WebDriverWait(panel_to_process_available, 10).until(
                        EC.presence_of_element_located((By.XPATH, difficulty_container_xpath))
                    )
                    difficulty_dropdown_selector = (By.CSS_SELECTOR, 'select.dropdown')
                    difficulty_dropdown = WebDriverWait(difficulty_container, 10).until(
                        EC.element_to_be_clickable(difficulty_dropdown_selector)
                    )
                    Select(difficulty_dropdown).select_by_value(DIFFICULTY_VALUE)
                    print(f"DEBUG: 槽 {slot_index} 難度已設定為 {DIFFICULTY_VALUE}")
                    time.sleep(random.uniform(0.5, 1.0))

                    print(f"DEBUG: 在槽位 {slot_index} 的交互面板內尋找備註輸入框...")
                    memo_input_selector = (By.CSS_SELECTOR, 'div.input-div--full input.standard-text-box')
                    memo_input = WebDriverWait(panel_to_process_available, 10).until(
                        EC.element_to_be_clickable(memo_input_selector)
                    )
                    memo_input.clear()
                    memo_input.send_keys(MEMO_TEXT)
                    print(f"DEBUG: 槽 {slot_index} 備註已設定為 '{MEMO_TEXT}'")
                    time.sleep(random.uniform(0.5, 1.0))

                    print(f"DEBUG: 在槽位 {slot_index} 的交互面板內尋找開始按鈕...")
                    start_button_selector = (By.CSS_SELECTOR, 'div.button-container button.button--start:not(.disabled)')
                    start_button = WebDriverWait(panel_to_process_available, 10).until(
                        EC.element_to_be_clickable(start_button_selector)
                    )
                    start_button.click()
                    print(f"DEBUG: 槽 {slot_index} 開始按鈕已點擊。")

                    print(f"DEBUG: 槽 {slot_index} 處理開始模擬確認對話框...")
                    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'div.modal-container')))
                    ok_button_global = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, 'button.ok-button')))
                    
                    try: 
                        WebDriverWait(driver, 3).until(EC.invisibility_of_element_located((By.CSS_SELECTOR, ".modal-overlay")))
                    except TimeoutException:
                        print("訊息：modal-overlay 在3秒內未消失。")
                    
                    driver.execute_script("arguments[0].scrollIntoView({block: 'center', inline: 'nearest'});", ok_button_global)
                    time.sleep(random.uniform(0.5, 1.0))
                    driver.execute_script("arguments[0].click();", ok_button_global)
                    print(f"DEBUG: 槽 {slot_index} 確認對話框 OK 已點擊。模擬應已啟動。")
                    
                    action_taken_in_current_scan = True
                    current_global_input_index += 1 
                    time.sleep(random.uniform(2, 4)) 
                
                except StaleElementReferenceException as e_stale_available:
                    print(f"錯誤: 在為槽 {slot_index} (Available) 與面板互動時發生 StaleElementReferenceException: {e_stale_available}")
                    print("將在下一輪嘗試。")
                    action_taken_in_current_scan = True # To trigger a quick rescan after refresh
                    current_global_input_index += 1 # Consume the input mapping attempt as it failed for this panel
                    # No `go_back_to_simulator` here, let the main loop refresh and retry.
                    # We don't break the inner loop here, to allow other slots to be processed if possible.
                    # However, if a Stale Ref occurs, it often means the page is unstable, so a break might be safer.
                    # For now, let's continue to see if other slots can be processed.
                    continue # Continue to the next slot_info

                except Exception as e_start:
                    print(f"啟動槽 {slot_index} 時發生錯誤: {e_start}")
                    action_taken_in_current_scan = True # To avoid immediate long wait
                    current_global_input_index += 1 # Consume the input mapping attempt
                    # Continue to the next slot_info if one action failed
                    continue
            
            elif current_status_text == "In Progress":
                print(f"槽 {slot_index} 正在進行中。跳過。")
                # This slot is busy, so we don't consider it "available" for starting new one
            
            else: # Unknown, Error, etc.
                print(f"槽 {slot_index} 狀態為 '{current_status_text}' (未處理)。跳過。")

            if success_count >= MAX_SUCCESS: break # Break from for-loop
        # --- End of for loop iterating through slots ---

        if success_count >= MAX_SUCCESS:
            print(f"已達到目標成功次數 ({MAX_SUCCESS})。流程結束。")
            break # Break from while-loop

        if action_taken_in_current_scan:
            print("本輪掃描已執行操作 (啟動新模擬或處理完成結果)。將立即開始下一輪掃描。")
            scans_with_no_action_count = 0 # Reset counter
            time.sleep(random.uniform(3,7)) # Shorter pause if action was taken
        else: # No action was taken in the entire scan of all slots
            scans_with_no_action_count +=1
            print(f"本輪掃描未執行任何操作。檢查是否所有槽都不可啟動...")
            if not at_least_one_slot_is_available : # if all slots were "In Progress", "Finished", "Error", etc.
                print(f"所有槽均非 'Available' 狀態。等待 60 秒... (無操作掃描次數: {scans_with_no_action_count})")
                time.sleep(60)
                print("60秒等待結束。刷新頁面...")
                driver.refresh()
                # After refresh, ensure page is ready before next scan
                try:
                    WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.XPATH, slot_panel_xpath)))
                    print("頁面刷新完成，模擬面板已載入。")
                except TimeoutException:
                    print("錯誤：刷新後等待模擬面板載入超時。流程可能中斷。")
                    break # Critical error, break main loop
                scans_with_no_action_count = 0 # Reset after long wait and refresh
            else: # Some slots were 'Available' but couldn't be started (e.g. error during start process) or other logic paths
                print(f"本輪未執行操作，但曾有 'Available' 的槽或未觸發長等待條件。等待 15 秒後重試... (無操作掃描次數: {scans_with_no_action_count})")
                time.sleep(15)
                if scans_with_no_action_count >= max_scan_retries_with_no_action:
                    print(f"連續 {max_scan_retries_with_no_action} 次掃描未執行有效操作。嘗試刷新頁面...")
                    driver.refresh()
                    try:
                        WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.XPATH, slot_panel_xpath)))
                        print("頁面刷新完成，模擬面板已載入。")
                    except TimeoutException:
                        print("錯誤：刷新後等待模擬面板載入超時。流程可能中斷。")
                        break
                    scans_with_no_action_count = 0


    print("模擬流程結束。")
    driver.quit()

if __name__ == "__main__":
    main()
