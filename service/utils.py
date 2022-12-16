def fake_hash_password(password: str):
    return "fakehashed" + password


MODEL_NAMES = [
    'homework_1',
    'user_knn',
    'most_popular',
    'light_fm'
]

fake_users_db = {
    "admin": {
        "username": "admin",
        "full_name": "Greatest Admin",
        "email": "admin@example.com",
        "hashed_password": "fakehashedqwerty1234",
        "disabled": False,
    }
}
