POSTGRES_INDEX_COMMANDS = """CREATE INDEX songs_fingerprinted ON songs ("fingerprinted");
 CREATE INDEX songs_date_modified ON songs ("date_modified");
 CREATE INDEX songs_file_sha1 ON songs ("file_sha1");
 CREATE INDEX songs_song_id ON songs ("song_id");
 CREATE INDEX fingerprints_offset ON fingerprints ("offset");
 CREATE INDEX songs_date_created ON songs ("date_created");
 CREATE INDEX fingerprints_date_created ON fingerprints ("date_created");
 CREATE INDEX fingerprints_hash ON fingerprints ("hash");
 CREATE INDEX fingerprints_song_id ON fingerprints ("song_id");
 CREATE INDEX songs_total_hashes ON songs ("total_hashes");
 CREATE INDEX songs_song_name ON songs ("song_name");
 CREATE INDEX fingerprints_date_modified ON fingerprints ("date_modified");"""


POSTGRES_INDEX_COMMANDS_LIST = ['CREATE INDEX songs_fingerprinted ON songs ("fingerprinted");',
 'CREATE INDEX songs_date_modified ON songs ("date_modified");',
 'CREATE INDEX songs_file_sha1 ON songs ("file_sha1");',
 'CREATE INDEX songs_song_id ON songs ("song_id");',
 'CREATE INDEX fingerprints_offset ON fingerprints ("offset");',
 'CREATE INDEX songs_date_created ON songs ("date_created");',
 'CREATE INDEX fingerprints_date_created ON fingerprints ("date_created");',
 'CREATE INDEX fingerprints_hash ON fingerprints ("hash");',
 'CREATE INDEX fingerprints_song_id ON fingerprints ("song_id");',
 'CREATE INDEX songs_total_hashes ON songs ("total_hashes");',
 'CREATE INDEX songs_song_name ON songs ("song_name");',
 'CREATE INDEX fingerprints_date_modified ON fingerprints ("date_modified");']


