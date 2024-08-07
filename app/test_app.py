"""Testing Module"""

import requests

BASE_URL = "http://localhost:8080"


def test_invalid_payload1():
    """
    test using invalid data: string instead of float as a feature value
    """
    payload = {
        "input_data": "abc, -1.593959186646871, -1.3028062187698115, -1.0835721055756011, 0.42981925801619825, -0.7470859700913761, -0.743747898186249, -0.7263365109480833, 0.012345193493854038, 0.8863413657382382, -0.46151710666317775, -0.4355394704137778, -0.47377361345186314, -0.5420578604224968, 0.8550416969399245, -0.6236232406930352, -0.3993337791104003, 0.3915518965745428, -0.03296952756533042, -0.31277749726245113, -1.2506107043148174, -1.631242605490714, -1.254913492614529, -0.9944215616996749, 0.001376708925086159, -0.8871925351653347, -0.8804342413449833, -0.7969025993114909, -0.7292238509850946, -0.34445459263314576"
    }
    response = requests.post(f"{BASE_URL}/predict", data=payload, timeout=5.0)
    assert response.status_code == 400


def test_invalid_payload2():
    """
    test using invalid data: not all features are passed
    """
    payload = {
        "input_data": "-1.313080487090047, -1.593959186646871, -1.3028062187698115, -1.0835721055756011, 0.42981925801619825, -0.7470859700913761, -0.743747898186249, -0.7263365109480833, 0.012345193493854038, 0.8863413657382382, -0.46151710666317775, -0.4355394704137778, -0.47377361345186314, -0.5420578604224968, 0.8550416969399245, -0.6236232406930352, -0.3993337791104003, 0.3915518965745428, -0.03296952756533042, -1.631242605490714, -1.254913492614529, -0.9944215616996749, 0.001376708925086159, -0.8871925351653347, -0.8804342413449833, -0.7969025993114909, -0.7292238509850946, -0.34445459263314576"
    }
    response = requests.post(f"{BASE_URL}/predict", data=payload, timeout=5.0)
    assert response.status_code == 400


def test_valid_payload1():
    """
    test using valid data for diagnosis type 'B'
    """
    payload = {
        "input_data": "0.46939260797037663, -0.325707602726287, 0.47908184355672984, 0.35867232980199026, 0.05264241567218082, 0.4711151264316021, 0.13484897953024655, 0.4421308885217207, 0.1109206652433763, -0.2803467735953654, 0.3631873829199002, -0.42084328484590433, 0.3455020472147773, 0.30412789231950216, -0.42334346761886776, 0.8457127509817742, -0.1320881742179115, 0.16608046142565783, -0.05597444324184531, 0.13204608104251317, 0.8595602763812112, 0.2610022511939757, 0.8709017955401637, 0.735540337319294, 0.31699512662490165, 1.9506267402998587, 0.5963874260628176, 1.0109513082435948, 1.4418377295066254, 1.15565158612946"
    }
    response = requests.post(f"{BASE_URL}/predict", data=payload, timeout=5.0)
    assert response.status_code == 200


def test_valid_payload2():
    """
    test using valid data for diagnosis type 'M'
    """
    payload = {
        "input_data": "-1.313080487090047, -1.593959186646871, -1.3028062187698115, -1.0835721055756011, 0.42981925801619825, -0.7470859700913761, -0.743747898186249, -0.7263365109480833, 0.012345193493854038, 0.8863413657382382, -0.46151710666317775, -0.4355394704137778, -0.47377361345186314, -0.5420578604224968, 0.8550416969399245, -0.6236232406930352, -0.3993337791104003, 0.3915518965745428, -0.03296952756533042, -0.31277749726245113, -1.2506107043148174, -1.631242605490714, -1.254913492614529, -0.9944215616996749, 0.001376708925086159, -0.8871925351653347, -0.8804342413449833, -0.7969025993114909, -0.7292238509850946, -0.34445459263314576"
    }
    response = requests.post(f"{BASE_URL}/predict", data=payload, timeout=5.0)
    assert response.status_code == 200
