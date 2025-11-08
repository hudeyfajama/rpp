import requests

url = "http://localhost:8001/predict"

payload = {
    "country_code": "SOM",
    "county": "Banadir",
    "subcounty": "Mogadishu",
    "latitude": 2.0469,
    "longitude": 45.3182,
    "commodity": "Maize",
    "price_flag": "Rice",
    "price_type": "Retail",
    "year": 2025,
    "month": 11,
    "day": 7
}

response = requests.post(url, json=payload)

print("Status Code:", response.status_code)
print("Response:", response.json())