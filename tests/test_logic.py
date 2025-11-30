# tests/test_logic.py

def test_dummy():
    # Test này luôn đúng, mục đích để kiểm tra xem hệ thống CI có chạy không
    assert 1 + 1 == 2

def test_data_structure():
    # Giả lập kiểm tra cấu trúc dữ liệu đầu vào
    input_data = {
        "features": {
            "tenure": 12,
            "monthlycharges": 50.5
        }
    }
    assert "features" in input_data
    assert isinstance(input_data["features"], dict)
    assert input_data["features"]["tenure"] == 12