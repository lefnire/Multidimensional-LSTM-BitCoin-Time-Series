import json
from sqlalchemy import create_engine
from box import Box

config = Box(json.loads(open('configs.json').read()))

engine = create_engine(config['data']['db_in']) #, echo=True
conn = engine.connect()