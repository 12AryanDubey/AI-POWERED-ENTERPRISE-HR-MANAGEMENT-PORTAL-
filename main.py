import sqlite3

# This creates a local database file named 'hr_database.db' in your project folder
try:
    conn = sqlite3.connect('hr_database.db')
    cursor = conn.cursor()
    
    # Create an employees table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS employees (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            department TEXT NOT NULL,
            performance_score REAL,
            attrition_risk REAL
        )
    ''')
    
    conn.commit()
    conn.close()
    
    print(" Connected to Local SQLite Database successfully!")
    print(" Created 'hr_database.db' in your project folder.")

except Exception as e:
    print(" Error:", e)