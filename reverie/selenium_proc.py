from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options  # => 引入Chrome的配置
from selenium.webdriver.common.by import By
import time
import os
import threading

# 配置
ch_options = Options()
ch_options.add_argument("--no-sandbox")  #
ch_options.add_argument("--window-size=1920,1080")  #
ch_options.add_argument("--start-maximized")
ch_options.add_argument("--headless=new")  # => 为Chrome配置无头模式
ch_options.add_argument('--disable-dev-shm-usage')  # 仅在docker里运行时用这个

# ch_options.binary_location =

# 在启动浏览器时加入配置
driver = webdriver.Chrome(options=ch_options) # => 注意这里的参数

target_link = 'http://localhost:8000/simulator_home'
# target_link = 'http://localhost:8000/replay/July1_the_ville_isabella_maria_klaus-step-3-20/1/'
sc_data_path = './data/bked_screenshot/tmp'

driver.get(target_link)
driver.save_screenshot(os.path.join(sc_data_path, './ch.png'))


quit_flag = False
pause_flag = False


def auto_screenshot():
    sc_count = 0
    time_wait_multiplier = 20
    ti = 0
    global quit_flag
    global pause_flag
    while not quit_flag:
        if pause_flag:
            time.sleep(1)
            continue
        if ti < time_wait_multiplier:
            ti += 1
        else:
            ti = 0
            driver.save_screenshot(os.path.join(sc_data_path, f'screenshot_{sc_count}.png'))
            sc_count += 1
        time.sleep(0.5)


t = threading.Thread(target=auto_screenshot)
t.start()

while True:
    signal = input('>>')
    if signal == 'q':
        break
    elif signal == 'p':
        pause_flag = True
    elif signal == 'c':
        pause_flag = False

quit_flag = True
t.join()
driver.quit()

