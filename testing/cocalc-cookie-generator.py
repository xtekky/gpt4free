import os
from selenium import webdriver
import random
import string

options = webdriver.FirefoxOptions()
options.add_argument('-headless')
options.add_argument("--disable-extensions")
options.add_argument("--disable-infobars")
options.add_argument("--disable-notifications")
options.add_argument("--disable-popup-blocking")
options.add_argument("--disable-session-crashed-bubble")
options.add_argument("--disable-geolocation")
options.add_argument("--disable-web-security")
options.add_argument("--disable-xss-auditor")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
service = webdriver.firefox.service.Service(executable_path='geckodriver')
driver = webdriver.Firefox(service=service, options=options)
driver.get("https://cocalc.com/")

# Ejecuta JavaScript para obtener todas las cookies almacenadas en la sesión
all_cookies = driver.execute_script("return document.cookie")

# Genera una cadena aleatoria de 10 caracteres
rand_str = ''.join(random.choices(string.ascii_letters + string.digits, k=10))

# Define la ruta absoluta de la carpeta donde se guardarán las cookies
script_dir = os.path.dirname(os.path.abspath(__file__))
cookies_dir = os.path.join(script_dir, 'cookies')

# Crea la carpeta si no existe
if not os.path.exists(cookies_dir):
    os.makedirs(cookies_dir)

# Crea el nombre del archivo con extensión .txt
filename = os.path.join(cookies_dir, f"{rand_str}.txt")

# Abre el archivo en modo escritura y escribe las cookies
with open(filename, "w", newline='') as f:
    f.write(all_cookies)

# Verifica si el archivo se escribió correctamente
if os.path.isfile(filename):
    print(f"Cookies guardadas en {filename}")
else:
    print("Error al guardar las cookies")

# Imprime todas las cookies en consola
print(all_cookies)

# Cierra el navegador
driver.quit()


# Imprime todas las cookies en consola
print(all_cookies)

# Cierra el navegador
driver.quit()
