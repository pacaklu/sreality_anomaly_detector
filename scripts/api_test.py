import requests
flat_id_to_test = [4065768524]

if __name__ == '__main__':
    for flat_id in [flat_id_to_test]:
        API_ENDPOINT = r'http://localhost:8000/predict?input_data=4065768524'
        r = requests.post(url=API_ENDPOINT)
