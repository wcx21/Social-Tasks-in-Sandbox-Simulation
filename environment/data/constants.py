import json
from datetime import datetime

TIME_CHOICE = [
    datetime(year=2023, month=2, day=14, hour=10),
    datetime(year=2023, month=2, day=14, hour=15),
    datetime(year=2023, month=2, day=14, hour=17),
    datetime(year=2023, month=2, day=14, hour=19),
    # 'on February 13, 2023 at 5pm',
    # 'on February 13, 2023 at 9pm',
    # 'on February 13, 2023 at 2pm',
]

ROLE_TO_NAME = {
    'cafe owner': ['Isabella Rodriguez'],
    'pharmacy shop keeper': ['John Lin'],
    'student': ['Ayesha Khan', 'Eddy Lin', 'Klaus Mueller', 'Maria Lopez', 'Wolfgang Schulz'],
    'artist': ['Abigail Chen', 'Francisco Lopez', 'Hailey Johnson', 'Jennifer Moore', 'Latoya Williams', 'Rajiv Patel'],
    'professor': ['Mei Lin'],
    'bartender': ['Arthur Burton'],
    'shopkeeper': ['Carmen Ortiz'],  # Harvey Oak Supply Store
    'grocery shop keeper': ['Tom Moreno'],

    'have_house': ['Adam Smith', 'Yuriko Yamamoto', 'Jennifer Moore', 'Sam Moore', 'Tamara Taylor', 'Carmen Ortiz'
                   'Eddy Lin', 'John Lin', 'Mei Lin'],

}

character_meta = json.load(open('./characters/characteristics.json', 'r', encoding='utf-8'))
