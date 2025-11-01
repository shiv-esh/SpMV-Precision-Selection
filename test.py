import requests

proxy_host = '172.31.2.4'
proxy_port = '8080'
proxy_username = 'your_proxy_username'  # Optional
proxy_password = 'your_proxy_password'  # Optional

proxy_url = f'http://{proxy_host}:{proxy_port}'
proxies = {
    'http': proxy_url,
    'https': proxy_url
}

try:
    response = requests.get('https://www.google.com', proxies=proxies)
    print("Proxy test successful:", response.status_code)
except Exception as e:
    print("Proxy test failed:", e)
